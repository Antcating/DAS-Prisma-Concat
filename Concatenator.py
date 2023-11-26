from config import DX, PRR, UNIT_SIZE, CHUNK_SIZE


import h5py
import numpy as np
from obspy import read
from obspy.core import Stream, Stats
from obspy.io.segy.segy import iread_segy

import datetime
import json
import os


class PrismaConcatenator:
    def __init__(self, raw_data_dir, target_folder):
        self.parent_input_dir = raw_data_dir
        self.parent_output_dir = target_folder

        self.dx = None
        self.gauge_m = None
        self.prr = None

        self.space_down_factor = None
        self.time_down_factor = None

    def str2arr(self, data, min_max_R_ind):
        """Converts data from obspy stream to numpy array"""
        return np.array(
            [
                data[i].data
                for i in np.arange(
                    min_max_R_ind[0], min_max_R_ind[1], self.space_down_factor
                )
            ]
        )

    
    def concat_arrays(self, array_from: np.ndarray, array_to: np.ndarray) -> np.ndarray:
        """Resizes and resizes h5py Dataset according to another h5py Dataset

        Args:
            dset_concat_from (Dataset): Donor Numpy array
            dset_concat_to (Dataset): Numpy array to be resized and appended to

        Returns:
            Dataset: Resized and appended dataset
        """
        if array_from.shape[0] == 0:
            return array_to
        # print(dset_concat_from.shape, dset_concat_to.shape)
        try:
            array_to.resize(
                array_to.shape[0], array_to.shape[1] + array_from.shape[1], refcheck=False
            )
            # Appending transposed data to the end of the Dataset
            # (Mekorot and Natgas datasets are transposed)
            array_to[:, -array_from.shape[1] :] = array_from[()]
            print(
                f"Concat successful, resulting dataset: {array_to.shape}"
            )
            return array_to
        except Exception as err:
            print(f"Critical error while saving last chunk: {err}")

    def read_Prisma_segy(
        self,
        files_sorted,
        min_max_R_ind=None,
    ):
        stats_first: Stats
        stats_last: Stats
        phase: np.ndarray
        distance_event: np.ndarray
        time_event: np.ndarray

        for file_index, file in enumerate(files_sorted):
            print("before reading", datetime.datetime.now())

            data = read(os.path.join(self.parent_input_dir, self.working_dir_r, file))
            print("after reading", datetime.datetime.now())

            if min_max_R_ind is None:
                min_max_R_ind = (0, len(data))

            if file_index == 0:
                stats_first: Stats = data[0].stats

            print("before processing", datetime.datetime.now())
            print("data read")
            phase_processing: np.ndarray = self.str2arr(data, min_max_R_ind)
            print("to np arr")

            if self.time_down_factor != 1:
                # downsample in time
                phase_processing = phase_processing.reshape(
                    phase_processing.shape[0], -1, self.time_down_factor
                ).mean(axis=-1)

            if file == files_sorted[0]:
                phase = phase_processing
            else:
                phase = self.concat_arrays(phase_processing, phase)
            print(phase.shape)
            print("after processing", datetime.datetime.now())
            if CHUNK_SIZE - (phase.shape[1] / PRR) < UNIT_SIZE:
                stats_last: Stats = data[0].stats
                packet_PRR = stats_last.sampling_rate
                packet_PRR_down = packet_PRR / self.time_down_factor
                packet_DX_down = self.dx * self.space_down_factor  # m
                distance_event: np.ndarray = (
                    np.arange(0, phase.shape[0] * packet_DX_down / 1000, packet_DX_down / 1000)
                    + min_max_R_ind[0] * self.dx / 1000
                )
                time_event: np.ndarray = np.arange(
                    0, phase.shape[1] / packet_PRR_down, 1.0 / packet_PRR_down
                )
                print(f"sample rate from {packet_PRR}Hz to {packet_PRR_down}Hz")
                print(f"dx from {self.dx}m to {packet_DX_down}m")
                
                del data
                return (
                    files_sorted[file_index + 1 :],
                    phase,
                    stats_first,
                    stats_last,
                    distance_event,
                    time_event,
                )
            
            del data

    def process_dir(self, working_dir_r):
        self.working_dir_r = working_dir_r
        # Iterate to find the working directory with SEGY files
        working_dir_list = os.listdir(
            os.path.join(self.parent_input_dir, self.working_dir_r)
        )
        while len(working_dir_list) == 1:
            self.working_dir_r = os.path.join(self.working_dir_r, working_dir_list[0])
            working_dir_list = os.listdir(
                os.path.join(self.parent_input_dir, self.working_dir_r)
            )

        json_file = [f for f in working_dir_list if f.endswith(".json")][0]
        print(json_file)

        ## read file ##
        files = [
            f
            for f in working_dir_list
            if not f.endswith(".json") and not f.endswith("Identifier")
        ]
        print(len(files), "data files")
        file_times = []
        for f in files:
            file_times.append(
                datetime.datetime.strptime(f.rstrip(".segy"), "%Y-%m-%dT%H-%M-%S-%f")
            )

        ## sort files by time ##
        sort_indeces = np.argsort(file_times)
        files_sorted = [files[i] for i in sort_indeces]
        times_sorted = [file_times[i] for i in sort_indeces]
        print(files_sorted)
        ## read metadata ##
        meta_file = open(
            os.path.join(self.parent_input_dir, self.working_dir_r, json_file)
        )
        meta = json.load(meta_file)
        print(meta)
        self.dx = meta["dx"]
        self.gauge_m = meta["gaugeLengthMeters"]
        self.prr = meta["prr"]
        print(self.dx, self.gauge_m)

        ## define downsampling factor ##
        # Time downsample factor
        if PRR > 0:
            self.time_down_factor = round(self.prr / PRR)
        else:
            self.time_down_factor = 1
        # Space downsample factor
        if DX > 0:
            self.space_down_factor = round(self.dx / DX)
        else:
            self.space_down_factor = 1

        ## read data ##
        min_max_R_ind = None
        
        while len(files_sorted) > 0:
            start_time = datetime.datetime.now()
            (
                files_sorted,
                phase,
                stats_first,
                stats_last,
                distance_event,
                time_event,
            ) = self.read_Prisma_segy(files_sorted, min_max_R_ind)
            print(
                "read from segy in %s seconds"
                % (datetime.datetime.now() - start_time).total_seconds()
            )

            h5_concat = os.path.join(
                self.parent_output_dir,
                f"{stats_first.starttime}_to_{stats_last.endtime}.hdf5",
            ).replace(":", "_")

            # save data to hdf5 file
            with h5py.File(h5_concat, "w") as f:
                f.create_dataset("phase_down", data=phase)
                f.create_dataset("distance_event", data=distance_event)
                f.create_dataset("time_event", data=time_event)
            # save attrs to json file
            with open(os.path.join(self.parent_output_dir, "attrs.json"), "w") as f:
                json.dump(
                    {
                        "DX": self.dx,
                        "gauge_m": self.gauge_m,
                        "PRR": self.prr,
                        "down_factor_time": self.time_down_factor,
                        "down_factor_space": self.space_down_factor,
                    },
                    f,
                )

            del phase, distance_event, time_event, stats_first, stats_last
    def run(self):
        dirs: list[str] = os.listdir(self.parent_input_dir)
        for working_dir_r in dirs:
            self.process_dir(working_dir_r)
