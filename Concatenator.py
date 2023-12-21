import h5py
import numpy as np
import pandas as pd

import datetime
import json
import os

from config import DX, PRR, CHUNK_SIZE
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
        save_data(distance_event, time_event, start_time):
            Save data to HDF5 and JSON files.
        process_dir(working_dir_r):
            Process the specified working directory.
        run():
            Runs the concatenation process.
    """

    def __init__(self, raw_data_dir, target_folder):
        self.parent_input_dir = raw_data_dir
        self.parent_output_dir = target_folder

        self.df_file_data = None
        self.current_dir = None
        self.previous_dir = None

        self.dx = None
        self.previous_dx = None
        self.gauge_m = None
        self.previous_gauge_m = None
        self.prr = None
        self.previous_prr = None

        self.channels = None
        self.previous_channels = None

        self.space_down_factor = None
        self.time_down_factor = None

        self.data_concat = None
        self.data_concat_offset = 0
        self.split_offset = None

        self.data_carry = None
        self.data_down = None

        self.chunk_start_time = None

    def concat_arrays(self) -> np.ndarray:
        """
        Concatenates the data array to the phase array.

        Args:
            None

        Returns:
            np.ndarray: The concatenated phase array.
        """
        if self.data_concat is None:
            self.data_concat = np.empty(
                (self.channels, int(CHUNK_SIZE * self.prr / self.time_down_factor)),
                dtype=np.dtype("f4"),
            )
        if self.data_carry is not None:
            self.chunk_start_time -= self.data_carry.shape[1] / PRR
            self.data_concat[
                :,
                self.data_concat_offset : self.data_concat_offset
                + self.data_carry.shape[1],
            ] = self.data_carry
            self.data_concat_offset = self.data_carry.shape[1]
            self.data_carry = None
            self.split_offset = None

        if not self.split_offset:
            self.data_concat[
                :,
                self.data_concat_offset : self.data_concat_offset + self.data.shape[1],
            ] = self.data
            self.data_concat_offset += self.data.shape[1]
        else:
            log.debug(f"Phase offset: {self.data_concat_offset}")
            log.debug(f"Split offset: {self.split_offset}")
            log.debug(f"Shape before: {self.data_concat.shape}")

            self.data_concat[
                :, self.data_concat_offset : self.data_concat_offset + self.split_offset
            ] = self.data[:, : self.split_offset]
            self.data_concat_offset += self.split_offset
            log.debug(f"Shape after: {self.data_concat.shape}")
            self.data_carry = self.data[:, self.split_offset :]

    def read_Prisma_segy(
        self,
    ):
        """
        Reads Prisma SEGY files and returns the necessary data.

        Args:
            None

        Returns:
            tuple: A tuple containing the following elements:
                - A list of remaining file names after processing.
                - The start time of the last processed file.
                - The end time of the last processed file.
                - An array representing the distance event.
                - An array representing the time event.
        """

        for index, row in self.df_file_data[self.index :].iterrows():
            self.index += 1
            if self.chunk_start_time is None:
                self.chunk_start_time = row["timestamps"]
            self.current_dir = row["dirs"]
            json_file = self.current_dir + "-info.json"
            if not os.path.exists(
                os.path.join(self.parent_input_dir, self.current_dir, json_file)
            ):
                json_file = [
                    x
                    for x in os.listdir(
                        os.path.join(self.parent_input_dir, self.current_dir)
                    )
                    if x.endswith(".json")
                ][0]
            ## read metadata ##
            if self.previous_dir is None or self.previous_dir != self.current_dir:
                log.info("Reading metadata")
                with open(
                    os.path.join(self.parent_input_dir, self.current_dir, json_file)
                ) as meta_file:
                    meta = json.load(meta_file)
                    self.dx = meta["dx"]
                    self.gauge_m = meta["gaugeLengthMeters"]
                    self.prr = meta["prr"]
                    self.channels = meta["numSamplesPerTrace"]
                    self.traces = meta["numTraces"]

                ## define downsampling factor ##
                # Time downsample factor
                if self.previous_prr is not None and self.previous_prr != self.prr:
                    raise ValueError("PRR changed")
                if self.previous_dx is not None and self.previous_dx != self.dx:
                    raise ValueError("DX changed")
                if (
                    self.previous_gauge_m is not None
                    and self.previous_gauge_m != self.gauge_m
                ):
                    raise ValueError("Gauge length changed")

                if PRR > 0:
                    self.time_down_factor = round(self.prr / PRR)
                else:
                    self.time_down_factor = 1
                # Space downsample factor
                if DX > 0:
                    self.space_down_factor = round(self.dx / DX)
                else:
                    self.space_down_factor = 1
                self.space_down_step = 1 / self.space_down_factor
                # read bytes (offset 3600) 3714-3715 from segy file to get trace size
                # read bytes (offset 3600) 3716-3717 from segy file to get sample rate
                # source: https://www.igw.uni-jena.de/igwmedia/geophysik/pdf/seg-y-trace-header-format.pdf
                with open(
                    os.path.join(self.parent_input_dir, self.current_dir, row["files"]),
                    "rb",
                ) as f:
                    f.seek(3714)
                    self.segy_trc_size, self.segy_trc_sample_rate = np.frombuffer(
                        f.read(4), dtype=np.int16
                    )
                    self.sampling_rate = (
                        1e6 / self.segy_trc_sample_rate
                    )  # convert from microseconds to seconds
                self.mmap_dtype = np.dtype(
                    # 240 bytes for the header, then the data
                    # source: https://www.igw.uni-jena.de/igwmedia/geophysik/pdf/seg-y-trace-header-format.pdf
                    [("headers", np.void, 240), ("data", "f4", self.segy_trc_size)]
                )
                if self.previous_channels != self.channels:
                    log.warning("Number of channels changed")
                    self.data_concat = np.empty(
                        (
                            self.channels,
                            int(CHUNK_SIZE * self.prr / self.time_down_factor),
                        ),
                        dtype=np.dtype("f4"),
                    )

            log.debug("before reading")
            file_path = os.path.join(
                self.parent_input_dir, self.current_dir, row["files"]
            )
            log.info(f"Reading {row['files']}")

            segy_data = np.memmap(
                file_path, dtype=self.mmap_dtype, mode="r", offset=3600
            )
            data = segy_data["data"]
            start_time = row["timestamps"]
            start_datetime = datetime.datetime.fromtimestamp(start_time)
            end_time = start_time + round(
                (self.segy_trc_size - 1) / self.sampling_rate, 1
            )
            end_datetime = datetime.datetime.fromtimestamp(end_time)
            unit_size = round(end_time - start_time, 0)
            log.debug("after reading")
            log.debug("before processing")

            if start_datetime.strftime("%d") != end_datetime.strftime("%d"):
                log.debug("Cutting data")
                cut_offset = (
                    end_datetime.replace(hour=0, minute=0, second=0) - start_datetime
                ).seconds
                self.split_offset = int(cut_offset * PRR)

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
            if self.space_down_factor != 1:
                # downsample in space
                self.data = self.data[:, :: self.space_down_step]

            if CHUNK_SIZE - (self.data_concat_offset / PRR) < unit_size:
                # ADD SPLITTING FOR USUAL CHUNKS TO NEXT DAY
                cut_offset = CHUNK_SIZE - (self.data_concat_offset / PRR)
                self.split_offset = int(cut_offset * PRR)
                log.info("Chunk size reached")

            self.concat_arrays()

            self.previous_dir = self.current_dir
            self.previous_dx = self.dx
            self.previous_gauge_m = self.gauge_m
            self.previous_prr = self.prr
            self.previous_channels = self.channels

            log.debug(self.data_concat.shape)
            log.debug(self.data_concat_offset)
            log.debug("after processing")

            if self.data_carry is not None:
                log.debug("Data carry to next chunk")
                log.debug(self.data_carry.shape)
                break

            elif self.data_concat_offset / PRR > CHUNK_SIZE:
                log.warning("Chunk size exceeded")
                break

            elif (
                self.index < len(self.df_file_data) - 1
                and round(self.time_diff[self.index], 0) > unit_size
            ):
                log.debug(f"Time diff: {self.time_diff[self.index]}")
                log.warning("Gap in data")
                break

        packet_PRR = self.sampling_rate
        packet_PRR_down = packet_PRR / self.time_down_factor
        packet_DX_down = self.dx * self.space_down_factor  # m

        log.info(f"Sample rate from {packet_PRR}Hz to {packet_PRR_down}Hz")
        log.info(f"Dx from {self.dx}m to {packet_DX_down}m")
        del data
        return (
            start_time,
            end_time,
        )

    def save_data(self):
        """
        Save data to HDF5 and JSON files.

        Args:
            distance_event (numpy.ndarray): Array of distance events.
            time_event (numpy.ndarray): Array of time events.
            start_time (datetime.datetime): Start time of the data.
            end_time (datetime.datetime): End time of the data.
        """
        data_start_date = datetime.datetime.fromtimestamp(self.chunk_start_time)
        date_start_str = data_start_date.strftime("%Y%m%d")
        save_path = os.path.join(
            self.parent_output_dir,
            str(data_start_date.year),
            date_start_str,
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        h5_concat = os.path.join(
            save_path,
            f"{date_start_str}_{str(self.chunk_start_time)}.h5",
        )
        log.info("Saving data to: " + h5_concat)
        log.debug(f"Final shape: {(self.channels, self.data_concat_offset)}")
        # save data to hdf5 file
        with h5py.File(h5_concat, "w") as f:
            f.create_dataset(
                "phase_down", data=self.data_concat[:, : self.data_concat_offset]
            )
        # Save last start time to file
        with open(
            os.path.join(self.parent_output_dir, "last_start"),
            "w",
        ) as f:
            f.write(str(self.chunk_start_time))
        # Save current dir to file
        self.save_last_dir()

        self.data_concat_offset = 0
        # save attrs to json file
        with open(os.path.join(save_path, "attrs.json"), "w") as f:
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

    def save_processed_dir(self, processed_dir):
        """
        Save processed directory to file.

        Args:
            None

        Returns:
            None
        """

        # If last row of file is not the same as previous dir, save it
        with open(os.path.join(self.parent_output_dir, "processed_dirs"), "a+") as f:
            f.write(processed_dir + "\n")

    def read_processed_dir(self):
        if os.path.exists(os.path.join(self.parent_output_dir, "processed_dirs")):
            with open(os.path.join(self.parent_output_dir, "processed_dirs"), "r") as f:
                processed_dirs = [x.strip() for x in f.readlines()]
                log.info(
                    "Dirs already processed:"
                    + " ".join([x[:5] + "..." for x in processed_dirs])
                )
                return processed_dirs
        else:
            return []

    def save_last_dir(self):
        """
        Save last directory to file.

        Args:
            None

        Returns:
            None
        """

        # If last row of file is not the same as previous dir, save it
        if os.path.exists(os.path.join(self.parent_output_dir, ".last_dir")):
            with open(os.path.join(self.parent_output_dir, ".last_dir"), "r") as f:
                last_dir = f.read()
                if last_dir != self.current_dir:
                    self.save_processed_dir(last_dir)
                    with open(
                        os.path.join(self.parent_output_dir, ".last_dir"), "w"
                    ) as f:
                        f.write(self.current_dir)
        else:
            with open(os.path.join(self.parent_output_dir, ".last_dir"), "w") as f:
                f.write(self.current_dir)

    def process_dir(self):
        """
        Process the specified working directory.

        Args:
            working_dir_r (str): The relative path of the working directory.

        Returns:
            None
        """

        self.index = 0
        while self.index < len(self.df_file_data):
            run_start_time = datetime.datetime.now()
            (
                start_time,
                end_time,
            ) = self.read_Prisma_segy()
            log.info(
                "read from segy chunk in %s seconds"
                % (datetime.datetime.now() - run_start_time).total_seconds()
            )
            self.save_data()  # save data to hdf5 file
            del start_time, end_time

            self.previous_dx = None
            self.previous_gauge_m = None
            self.previous_prr = None

            self.chunk_start_time = None

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
        processed_dirs = self.read_processed_dir()
        dirs = [d for d in dirs if d not in processed_dirs]

        # if directory last modified less than 2 day ago, skip it
        dirs = [
            d
            for d in dirs
            # if (
            #     datetime.datetime.now()
            #     - datetime.datetime.fromtimestamp(
            #         os.path.getmtime(os.path.join(self.parent_input_dir, d))
            #     )
            # ).days
            # >= 2
        ]
        log.info("Skipped dirs modified less than 2 days ago")

        self.df_file_data = pd.DataFrame(columns=["dirs", "files", "timestamps"])

        # Read last start time
        if os.path.exists(os.path.join(self.parent_output_dir, "last_start")):
            with open(os.path.join(self.parent_output_dir, "last_start"), "r") as f:
                last_start_time = float(f.read())
                self.data_concat = h5py.File(
                    os.path.join(
                        self.parent_output_dir,
                        datetime.datetime.fromtimestamp(last_start_time).strftime(
                            "%Y/%Y%m%d"
                        ),
                        f"{datetime.datetime.fromtimestamp(last_start_time).strftime('%Y%m%d')}_{str(last_start_time)}.h5",
                    ),
                )["phase_down"][:]
                self.chunk_start_time = last_start_time
                self.data_concat_offset = self.data_concat.shape[1]
        else:
            last_start_time = 0

        for dir in dirs:
            files = os.listdir(os.path.join(self.parent_input_dir, dir))
            files = [f for f in files if not f.endswith(".json")]
            file_timestamps = []

            for f in files:
                file_time_str = f.rstrip(".segy")

                if file_time_str.endswith("Z"):
                    file_time_str = file_time_str[:-1]

                file_timestamps.append(
                    datetime.datetime.strptime(
                        file_time_str, "%Y-%m-%dT%H-%M-%S-%f"
                    ).timestamp()
                )

            self.df_file_data = pd.concat(
                [
                    self.df_file_data,
                    pd.DataFrame(
                        {
                            "dirs": [dir] * len(files),
                            "files": files,
                            "timestamps": file_timestamps,
                        }
                    ),
                ]
            )
        self.df_file_data.sort_values(by=["timestamps"], inplace=True)

        self.time_diff = np.diff(
            np.concatenate(([last_start_time], self.df_file_data["timestamps"].values))
        )
        self.process_dir()
        if self.current_dir is not None:
            self.save_processed_dir(self.current_dir)
        log.info("Concatenation finished")
