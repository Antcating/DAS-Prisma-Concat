import h5py
import numpy as np

import datetime
import json
import os

from config import DX, PRR, UNIT_SIZE, CHUNK_SIZE
from log.main_logger import logger as log


class PrismaConcatenator:
    """
    A class for concatenating Prisma SEGY files.

    Attributes:
        parent_input_dir (str): The parent input directory.
        parent_output_dir (str): The parent output directory.
        dx (float): The distance increment.
        gauge_m (float): The gauge length in meters.
        prr (float): The pulse repetition rate.
        space_down_factor (int): The space downsample factor.
        time_down_factor (int): The time downsample factor.
        phase (np.ndarray): The phase array.
        phase_offset (int): The phase offset.
        data_down (np.ndarray): The downsampled data array.

    Methods:
        concat_arrays(min_max_R_ind) -> np.ndarray:
            Concatenates the data array to the phase array.
        read_Prisma_segy(files_sorted, min_max_R_ind=None) -> tuple:
            Reads Prisma SEGY files and returns the necessary data.
        save_data(distance_event, time_event, start_time, end_time):
            Save data to HDF5 and JSON files.
        process_dir(working_dir_r):
            Process the specified working directory.
        run():
            Runs the concatenation process.
    """
    
    def __init__(self, raw_data_dir, target_folder):
        self.parent_input_dir = raw_data_dir
        self.parent_output_dir = target_folder

        self.dx = None
        self.gauge_m = None
        self.prr = None

        self.space_down_factor = None
        self.time_down_factor = None

        self.phase = None
        self.phase_offset = 0

        self.data_down = None

    def concat_arrays(self, min_max_R_ind) -> np.ndarray:
        """
        Concatenates the data array to the phase array.

        Args:
            min_max_R_ind (tuple): A tuple containing the minimum and maximum indices.

        Returns:
            np.ndarray: The concatenated phase array.
        """
        if self.phase is None:
            self.phase = np.empty(
                (min_max_R_ind[1] - min_max_R_ind[0], int(CHUNK_SIZE * PRR)),
                dtype=np.dtype("f4"),
            )

        # write data from array_from to array_to on the offset self.phase_offset
        self.phase[
            :, self.phase_offset : self.phase_offset + self.data.shape[1]
        ] = self.data
        self.phase_offset += self.data.shape[1]

    def read_Prisma_segy(
        self,
        files_sorted,
        min_max_R_ind=None,
    ):
        """
        Reads Prisma SEGY files and returns the necessary data.

        Args:
            files_sorted (list): A list of sorted SEGY file names.
            min_max_R_ind (tuple, optional): A tuple representing the minimum and maximum R index. Defaults to None.

        Returns:
            tuple: A tuple containing the following elements:
                - A list of remaining file names after processing.
                - The start time of the last processed file.
                - The end time of the last processed file.
                - An array representing the distance event.
                - An array representing the time event.
        """
        dtype = "f4"

        distance_event: np.ndarray
        time_event: np.ndarray

        last_endtime = None

        segy_trc_size = None
        segy_trc_sample_rate = None

        for file_index, file in enumerate(files_sorted):
            log.debug("before reading")
            file_path = os.path.join(self.parent_input_dir, self.working_dir_r, file)
            log.info(f"Reading {file}")
            if not segy_trc_size or not segy_trc_sample_rate:
                # read bytes 3714-3715 from segy file to get trace size
                # read bytes 3716-3717 from segy file to get sample rate
                with open(file_path, "rb") as f:
                    f.seek(3714)
                    segy_trc_size, segy_trc_sample_rate = np.frombuffer(
                        f.read(4), dtype=np.int16
                    )
                    sampling_rate = (
                        1e6 / segy_trc_sample_rate
                    )  # convert from microseconds to seconds
            mmap_dtype = np.dtype(
                [("headers", np.void, 240), ("data", dtype, segy_trc_size)]
            )
            segy_data = np.memmap(file_path, dtype=mmap_dtype, mode="r", offset=3600)
            data = segy_data["data"]

            start_time = datetime.datetime.strptime(
                file.rstrip(".segy"), "%Y-%m-%dT%H-%M-%S-%f"
            )
            end_time = start_time + datetime.timedelta(
                seconds=(segy_trc_size - 1) / sampling_rate
            )
            # data = read(os.path.join(self.parent_input_dir, self.working_dir_r, file))
            log.debug("after reading")

            if min_max_R_ind is None:
                min_max_R_ind = (0, len(data))

            if last_endtime is not None and (start_time - last_endtime).seconds > 1:
                log.warning(f"Gap in data: {start_time - last_endtime} seconds")
                break
            log.debug("before processing")
            # phase_processing: np.ndarray = self.str2arr(data, min_max_R_ind)

            if self.time_down_factor != 1:
                # downsample in time
                data = data.reshape(data.shape[0], -1, self.time_down_factor)
                if self.data_down is None or self.data_down.shape != data.shape[:2]:
                    self.data_down = np.empty(
                        (data.shape[0], data.shape[1]),
                        dtype=np.dtype("f4"),
                    )
                self.data = data.mean(
                    axis=-1,
                )

            self.concat_arrays(min_max_R_ind)

            last_endtime = end_time
            log.debug(self.phase.shape)
            log.debug("after processing")

            if CHUNK_SIZE - (self.phase_offset / PRR) < UNIT_SIZE:
                log.info("Chunk size reached")
                break
            elif self.phase_offset / PRR > CHUNK_SIZE:
                log.warning("Chunk size exceeded")
                break

        packet_PRR = sampling_rate
        packet_PRR_down = packet_PRR / self.time_down_factor
        packet_DX_down = self.dx * self.space_down_factor  # m
        distance_event: np.ndarray = (
            np.arange(
                0, self.phase.shape[0] * packet_DX_down / 1000, packet_DX_down / 1000
            )
            + min_max_R_ind[0] * self.dx / 1000
        )
        time_event: np.ndarray = np.arange(
            0, self.phase.shape[1] / packet_PRR_down, 1.0 / packet_PRR_down
        )
        log.info(f"Sample rate from {packet_PRR}Hz to {packet_PRR_down}Hz")
        log.info(f"Dx from {self.dx}m to {packet_DX_down}m")
        del data
        return (
            files_sorted[file_index + 1 :],
            start_time,
            end_time,
            distance_event,
            time_event,
        )

    def save_data(self, distance_event, time_event, start_time, end_time):
        """
        Save data to HDF5 and JSON files.

        Args:
            distance_event (numpy.ndarray): Array of distance events.
            time_event (numpy.ndarray): Array of time events.
            start_time (datetime.datetime): Start time of the data.
            end_time (datetime.datetime): End time of the data.
        """
        data_start_date = start_time.strftime("%Y%m%d")
        if not os.path.exists(os.path.join(self.parent_output_dir, data_start_date)):
            os.mkdir(os.path.join(self.parent_output_dir, data_start_date))
        h5_concat = os.path.join(
            self.parent_output_dir,
            data_start_date,
            f"{start_time}_to_{end_time}.hdf5",
        ).replace(":", "_")

        # save data to hdf5 file
        with h5py.File(h5_concat, "w") as f:
            f.create_dataset("phase_down", data=self.phase[:, : self.phase_offset])
            f.create_dataset("distance_event", data=distance_event)
            f.create_dataset("time_event", data=time_event)
        self.phase_offset = 0
        # save attrs to json file
        with open(
            os.path.join(self.parent_output_dir, data_start_date, "attrs.json"), "w"
        ) as f:
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

    def process_dir(self, working_dir_r):
        """
        Process the specified working directory.

        Args:
            working_dir_r (str): The relative path of the working directory.

        Returns:
            None
        """
        log.info(f"Processing {working_dir_r}")
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
        log.debug(f"Working directory: {self.working_dir_r}")
        json_file = [f for f in working_dir_list if f.endswith(".json")][0]

        ## read file ##
        files = [
            f
            for f in working_dir_list
            if not f.endswith(".json") and not f.endswith("Identifier")
        ]
        file_times = []
        for f in files:
            file_times.append(
                datetime.datetime.strptime(f.rstrip(".segy"), "%Y-%m-%dT%H-%M-%S-%f")
            )

        ## sort files by time ##
        sort_indeces = np.argsort(file_times)
        files_sorted = [files[i] for i in sort_indeces]
        log.debug(f"Num files sorted: {len(files_sorted)}")
        ## read metadata ##
        meta_file = open(
            os.path.join(self.parent_input_dir, self.working_dir_r, json_file)
        )
        meta = json.load(meta_file)
        self.dx = meta["dx"]
        self.gauge_m = meta["gaugeLengthMeters"]
        self.prr = meta["prr"]

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

        min_max_R_ind = None
        while len(files_sorted) > 0:
            run_start_time = datetime.datetime.now()
            (
                files_sorted,
                start_time,
                end_time,
                distance_event,
                time_event,
            ) = self.read_Prisma_segy(files_sorted, min_max_R_ind)
            log.info(
                "read from segy chunk in %s seconds"
                % (datetime.datetime.now() - run_start_time).total_seconds()
            )

            self.save_data(
                distance_event, time_event, start_time, end_time
            )  # save data to hdf5 file
            del distance_event, time_event, start_time, end_time

    def run(self):
        """
        Runs the concatenation process.

        This method performs the concatenation process by iterating over the directories,
        skipping already processed directories and directories modified less than 1 day ago.
        It calls the `process_dir` method for each working directory and saves the processed
        directories in a file.

        Returns:
            None
        """
        log.info("Starting concatenation")
        dirs: list[str] = os.listdir(self.parent_input_dir)
        # remove already processed dirs
        if os.path.exists(os.path.join(self.parent_output_dir, "processed_dirs.txt")):
            with open(
                os.path.join(self.parent_output_dir, "processed_dirs.txt"), "r"
            ) as f:
                processed_dirs = [x.strip() for x in f.readlines()]
                log.info(
                    "Dirs already processed:"
                    + " ".join([x[:5] + "..." for x in processed_dirs])
                )
            dirs = [d for d in dirs if d not in processed_dirs]
        # if directory last modified less than 1 day ago, skip it
        dirs = [
            d
            for d in dirs
            if (
                datetime.datetime.now()
                - datetime.datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(self.parent_input_dir, d))
                )
            ).days
            > 1
        ]
        log.info("Skipped dirs modified less than 1 day ago")

        for working_dir_r in dirs:
            log.info("Processing dir", working_dir_r)
            self.process_dir(working_dir_r)
            log.info("after processing dir", working_dir_r)
            # save processed dirs
            with open(
                os.path.join(self.parent_output_dir, "processed_dirs.txt"), "a"
            ) as f:
                f.write(working_dir_r + "\n")
        log.info("Concatenation finished")