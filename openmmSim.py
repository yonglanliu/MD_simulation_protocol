"""
OpenMM simulation Protocal 
Version 8.0.0
Created by: Yonglan Liu
Date: 04/26/2024
"""
from openmm.app import *
from openmm import *
from openmm.unit import *
import argparse
import os
import glob

########################################################################
#                       Minimization + MD simulation                   #              
########################################################################
class simJob:
    def __init__(self, PDB, PSF, PARAM, OPT, TEMP, BOX_SIZE, BOX_SHAPE="cubic", DIR="./", ENSEMBLE="NPT", RESTART_METHOD="state", N_GPU=3, TOTAL_SIMULATION_TIME=3):
        self.DIR = DIR

        # Simulation system basic information
        self.OPT = OPT
        self.box_size = BOX_SIZE
        self.box_shape = BOX_SHAPE
        self.TEMP = TEMP
        self.PRESSURE = 1.01325 * bar
        self.FRICTION = 1.0 / picoseconds # langevin damping coefficient of 1/ps
        self.TIMESTEP = 2 * femtoseconds  # integration step, fs 
        self.ENSEMBLE = ENSEMBLE  # "NVT", "NPT"
        self.N_GPU = N_GPU
        self.RESTART_METHOD = RESTART_METHOD # "state or checkpnt"

        # Define intervals for saving data
        self.log_interval = 10000  
        self.checkpoint_interval = 250000  # Save checkpoint every 50 ns (25,000,000 steps)
        self.trajectory_interval = 25000000  # Save trajectories to a seprarted file every 50 ns (25,000,000 steps)
        self.dcd_interval = 50000            # Save trj inverval to dcd 

        # -----------
        #   testing 
        # -----------
        # self.log_interval = 1000  
        # self.checkpoint_interval = 1000  
        # self.trajectory_interval = 10000  
        # self.dcd_interval = 500 

        assert self.checkpoint_interval <= self.trajectory_interval
        assert self.trajectory_interval % self.checkpoint_interval == 0
        assert self.checkpoint_interval >= self.dcd_interval
        assert self.checkpoint_interval % self.dcd_interval == 0 

        self.total_steps = int((TOTAL_SIMULATION_TIME * (10 ** 9)) / self.TIMESTEP.value_in_unit(femtoseconds))

        # Load molecular structure
        self.pdb = PDBFile(PDB)
        self.psf = CharmmPsfFile(PSF)

        #  Force Fields   		   
        self.params = CharmmParameterSet(PARAM)

        # MD integrator + langevin temperature control  
        self.integrator = LangevinMiddleIntegrator(self.TEMP, self.FRICTION, self.TIMESTEP)

        # Monte Carlo pressure control
        self.barostat = MonteCarloBarostat(self.PRESSURE, self.TEMP, 25)

    def run(self):
        # Reset pbc compatible for openmm
        self.pbcProcessing(self.pdb, self.psf, self.box_size, self.box_shape)

        # Construct modeller
        modeller = Modeller(self.psf.topology, self.pdb.positions)

        # Create OpenMM System
        self.system = self.psf.createSystem(self.params, nonbondedMethod=PME, nonbondedCutoff=1.2*nanometer, constraints=HBonds)

        # Monte Carlo pressure control. Turn it off for minization stage because mini needs short NVT run first
        if self.OPT == "md" and self.ENSEMBLE == "NPT":
             self.system.addForce(self.barostat)

        # Set platform: GPU or CPU
        try:
            PLATFORM = "GPU"
            platform = Platform.getPlatformByName('CUDA')
            properties = {'CudaDeviceIndex': ', '.join(str(i) for i in range(self.N_GPU)), 'CudaPrecision': 'single'} # must use single on Biowulf
            simulation = Simulation(modeller.topology, self.system, self.integrator, platform, properties)
        except:
            PLATFORM = "CPU"
            simulation = Simulation(modeller.topology, self.system, self.integrator)

        print(f"Simultation is performed on {PLATFORM}")

        # Set initial positions and velocities
        simulation.context.setPositions(modeller.positions)

        #----------------------
        #  Energy minimization      
        #----------------------
        if self.OPT == "mini":
            # Set up reporters
            dcdReporter = DCDReporter("traj/mini_0.dcd", 200)
            simulation.reporters.append(dcdReporter)
            state_file = os.path.join(self.DIR, f'output/data_0.txt')
            state_file = open(state_file, 'a')
            stateReporter = self.logger(state_file, 200, 0)
            simulation.reporters.append(stateReporter) 

            print("Minimization Energy")
            simulation.minimizeEnergy()
            
            # NVT short run
            print("Running NVT") 
            simulation.step(1000) 
            print(f"Short NVT simulation was finished. Current step: {simulation.currentStep}")

            # NPT short run 
            self.system.addForce(self.barostat)
            simulation.context.reinitialize(preserveState=True)
            print("Running NPT")
            simulation.step(1000) 
            print(f"Short NPT simulation was finished. Current step: {simulation.currentStep}")

            # save states
            print('Saving...')
            positions = simulation.context.getState(enforcePeriodicBox=True, getPositions=True).getPositions()
            PDBFile.writeFile(simulation.topology, positions, open(f'./mini.pdb', 'w'))
            simulation.saveCheckpoint(os.path.join(self.DIR, f'chk/check_mini.chk'))
            simulation.saveState(os.path.join(self.DIR, f'state/state_mini.xml'))
            print("Minimization has been successfully finished")

            simulation.reporters.remove(dcdReporter)
            simulation.reporters.remove(stateReporter)
            
        # ---------------------
        #     MD simulation   
        # ---------------------
        elif self.OPT == "md":
            # Restart simulation
            simulation, miniSteps = self.restart(simulation, self.RESTART_METHOD)

            dcd_path = lambda index: os.path.join(self.DIR, f'traj/md_{index}.dcd')  # Lambda function

            # --------------------------
            #    Main simulation loop
            # --------------------------
            currentStep = simulation.currentStep
            print(f"Loading state at {currentStep} step")
            totalSteps = self.total_steps + miniSteps
            print(f"Total steps is {totalSteps}")
            if currentStep == totalSteps:
                print("Already finished the simulations. If you want to continue to run, please increase the simulation time: TOTAL_SIMULATION_TIME (microseconds)")

            while currentStep < totalSteps:
                # Check if last simulation was finished or not
                reminder = (currentStep - miniSteps) % self.trajectory_interval 
                simIndex = (currentStep - miniSteps) // self.trajectory_interval + 1
                print(f"Current DCD index is {simIndex}")
                print(f'{reminder} steps are left for current DCD')

                # ----------------------------
                # DCD output and Setup logger
                # ----------------------------
                state_file = os.path.join(self.DIR, f'output/data_{simIndex}.txt')
                state_file = open(state_file, 'a')

                dcd_file = dcd_path(simIndex)

                if reminder > 0: 
                    dcdReporter = DCDReporter(dcd_file, self.dcd_interval, append=True)
                    steps_to_run = min(self.trajectory_interval - reminder, totalSteps - currentStep)
                else:
                    dcdReporter = DCDReporter(dcd_file, self.dcd_interval, append=False)
                    steps_to_run = min(self.trajectory_interval, totalSteps - currentStep)

                stateReporter = self.logger(state_file, self.log_interval, miniSteps)
                simulation.reporters.append(dcdReporter)
                simulation.reporters.append(stateReporter)
                
                # Run simulations
                while simulation.currentStep < currentStep + steps_to_run:
                    simulation.step(self.dcd_interval) 
                    simulation.saveCheckpoint(os.path.join(self.DIR, f"chk/check_md_latest.chk"))
                    simulation.saveState(os.path.join(self.DIR, f"state/state_md_latest.xml"))
                
                # Update current steps
                currentStep = simulation.currentStep

                # Save checkpoint and state
                simulation.saveCheckpoint(os.path.join(self.DIR, f"chk/check_md_{currentStep - miniSteps}.chk"))
                simulation.saveState(os.path.join(self.DIR, f"state/state_md_{currentStep - miniSteps}.xml"))


                # Final save
                if currentStep == totalSteps:
                    simulation.saveCheckpoint(os.path.join(self.DIR, f"chk/check_md_latest.chk"))
                    simulation.saveCheckpoint(os.path.join(self.DIR, f"chk/check_md_{currentStep - miniSteps}.chk"))
                    simulation.saveState(os.path.join(self.DIR, f"state/state_md_latest.xml"))
                    simulation.saveState(os.path.join(self.DIR, f"state/state_md_{currentStep - miniSteps}.xml"))

                # remove old dcd and sate reporters
                simulation.reporters.remove(dcdReporter)
                simulation.reporters.remove(stateReporter)

                print(f"MD simulations at {currentStep - miniSteps} has been successfully finished")

        else:
            raise ValueError("Unrecognized execution: {self.OPT}")

    def pbcProcessing(self, pdb, psf, box_size, shape):
        """
        if the unit cell vector's origin is at 0, 0, 0,
        Shift to box_size/2, box_size/2, box_size/2. Why? 
        which means that the center of the box is at box_size/2, box_size/2, box_size/2 for openMM. 
        ref https://github.com/openmm/openmm/issues/1390
        """
        assert shape in ["cubic", "rectangle"]
        assert isinstance(box_size, (list, tuple, int, float))

        if shape == "cubic" and isinstance(box_size, (int, float)):
            x, y, z = box_size, box_size, box_size
        else:
            assert len(box_size) == 3
            x, y, z = box_size
            assert isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float))

        # Calculate the center of geometry 
        import numpy as np
        positions = pdb.positions
        center_of_geometry = np.mean(positions.value_in_unit(nanometers), axis=0) * nanometers 
        
        # Reset box center at (x/2, y/2, z/2)
        box_center = Vec3(x/2, y/2, z/2)  
        shift = box_center * nanometers - center_of_geometry

        # update position
        new_positions = [pos + shift for pos in positions]
        pdb.positions = new_positions
        psf.setBox(x * nanometers, y * nanometers, z * nanometers)
    
    def logger(self, state_file, interval, miniSteps):
        # Setup the logger
        return app.StateDataReporter(state_file, interval, step=True, potentialEnergy=True, 
            totalEnergy=True, temperature=True, volume=True, progress=True, remainingTime=True, 
            speed=True, totalSteps=self.total_steps + miniSteps, separator='\t', append=True)

    def restart(self, simulation, method):
        # Obtain MD steps from minimization
        if method == "state":
            simulation.loadState('state/state_mini.xml')
            miniSteps = simulation.currentStep
            latest_file = os.path.join(self.DIR, f'state/state_md_latest.xml')

            if os.path.exists(latest_file): 
                simulation.loadState(latest_file)

        elif method == "checkpnt":
            simulation.loadCheckpoint('chk/check_mini.chk')
            miniSteps = simulation.currentStep
            latest_file = os.path.join(self.DIR, f'chk/check_md_latest.chk')

            if os.path.exists(latest_file): 
                simulation.loadCheckpoint(latest_file)
        else:
            raise ValueError("2nd variable should should be either state or checkpnt")

        return simulation, miniSteps



def main(args):
    simJOB = simJob(PDB=args.pdb, 
                    PSF=args.psf, 
                    PARAM="./par_all36.prm",  # path for parmeter file
                    OPT = args.opt, 
                    TEMP = args.temperature * kelvin, 
                    BOX_SIZE = args.boxSize, 
                    DIR = "./",
                    ENSEMBLE = "NPT", 
                    N_GPU=args.n_gpu, 
                    TOTAL_SIMULATION_TIME=args.simTime)
    simJOB.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=str, help="PDB file path")
    parser.add_argument("--psf", type=str, help="PSF file path")
    parser.add_argument("--opt", type=str, help="mini or md")
    parser.add_argument("--temperature", type=int, default=310)
    parser.add_argument("--boxSize", type=float, help='3D tuple, list or 1 float or integer')
    parser.add_argument("--n_gpu", type=int, default=3)
    parser.add_argument("--simTime", type=int, help="Simulation time (microseconds)")
    args = parser.parse_args()
    main(args)
