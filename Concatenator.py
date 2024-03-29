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

        Class for concatenating Prisma data arrays.

    Attributes:
        parent_input_dir (str): The parent input directory.
        parent_output_dir (str): The parent output directory.
        df_file_data (pandas.DataFrame): Files names and times dataframe.
        current_dir (str): The current directory.
        previous_dir (str): The previous directory.
        dx (float): The dx value.
        previous_dx (float): The previous dx value.
        gauge_m (float): The gauge length in meters.
        previous_gauge_m (float): The previous gauge length in meters.
        prr (float): The prr value.
        previous_prr (float): The previous prr value.
        channels (int): The number of channels.
        previous_channels (int): The previous number of channels.
        space_down_factor (float): The space downsample factor.
        time_down_factor (float): The time downsample factor.
        data_concat (numpy.ndarray): The concatenated data array.
        data_concat_offset (int): The offset of the concatenated data array.
        split_offset (int): The split offset.
        data_carry (numpy.ndarray): The data carry array.
        data_down (numpy.ndarray): The downsampled data array.
        chunk_start_time (float): The start time of the chunk.

    Methods:
        concat_arrays: Concatenates the data array to the phase array.
        read_Prisma_segy: Reads Prisma SEGY files and returns the necessary data.
        save_data: Save data to HDF5 and JSON files.
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

        self.dir_index = 0

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
                (self.channels, int(CHUNK_SIZE * PRR)),
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
            log.debug("Split chunk concatenated")
            self.data_carry = self.data[:, self.split_offset :]
            log.debug("Carry created")

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

        for row_index, row in self.df_file_data[self.index :].iterrows():
            self.index += 1
            if self.chunk_start_time is None:
                self.chunk_start_time = row["timestamps"]
            self.current_dir = row["dirs"]
            # Expected json file name
            json_file = self.current_dir + "-info.json"
            # If json file does not exist, get the first json file in the directory
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
            if self.previous_dir is None or self.previous_dir != self.current_dir:
                log.info("Reading metadata")
                with open(
                    os.path.join(self.parent_input_dir, self.current_dir, json_file)
                ) as attrs_json:
                    attrs = json.load(attrs_json)
                    self.dx = attrs["dx"]
                    self.gauge_m = attrs["gaugeLengthMeters"]
                    self.prr = attrs["prr"]
                    self.channels = attrs["numSamplesPerTrace"]
                # read bytes (offset 3600) 3714-3715 from segy file to get trace size

                with open(
                    os.path.join(self.parent_input_dir, self.current_dir, row["files"]),
                    "rb",
                ) as f:
                    f.seek(3714)
                    self.traces = np.frombuffer(f.read(2), dtype=np.int16)[0]

                # Time downsample factor
                if self.previous_prr is not None and self.previous_prr != self.prr:
                    self.dir_index += 1
                    log.warning("PRR changed")
                    break
                if self.prr != 1500:
                    raise ValueError("PRR is not 1500")
                if self.previous_dx is not None and self.previous_dx != self.dx:
                    self.dir_index += 1
                    log.warning("DX changed")
                    break
                if (
                    self.previous_gauge_m is not None
                    and self.previous_gauge_m != self.gauge_m
                ):
                    self.dir_index += 1
                    log.warning("Gauge length changed")
                    break

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
                self.mmap_dtype = np.dtype(
                    # 240 bytes for the header, then the data
                    # source: https://www.igw.uni-jena.de/igwmedia/geophysik/pdf/seg-y-trace-header-format.pdf
                    [("headers", np.void, 240), ("data", "f4", self.traces)]
                )
                if (
                    self.previous_channels is not None
                    and self.previous_channels != self.channels
                ):
                    self.dir_index += 1
                    log.warning("Number of channels changed")
                    if self.data_concat is not None:
                        self.save_data()
                        self.chunk_start_time = row["timestamps"]
                    self.data_concat = np.empty(
                        (
                            self.channels,
                            int(CHUNK_SIZE * PRR),
                        ),
                        dtype=np.dtype("f4"),
                    )

            log.debug("before reading")
            row_file_path = os.path.join(
                self.parent_input_dir, self.current_dir, row["files"]
            )
            log.info(f"Reading {row['files']}")
            try:
                segy_data = np.memmap(
                    row_file_path, dtype=self.mmap_dtype, mode="r", offset=3600
                )
            except ValueError:
                log.error(f"Error reading {row['files']}")
                break
            data = segy_data["data"]
            if data.shape[0] != self.channels:
                log.warning("Inside dir: Number of channels changed")
                self.dir_index += 1
                break
            start_time = row["timestamps"]
            start_datetime = datetime.datetime.fromtimestamp(start_time)
            end_time = start_time + round((self.traces - 1) / self.prr, 1)
            end_datetime = datetime.datetime.fromtimestamp(end_time)
            unit_size = self.traces / self.prr
            log.debug("after reading")
            log.debug("before processing")
            # It is possible to make more efficient, but LOCAL ISRAEL TIME IS NOT UTC
            if start_datetime.strftime("%d") != end_datetime.strftime("%d"):
                self.dir_index = 0
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

            if CHUNK_SIZE - (self.data_concat_offset / PRR) <= unit_size:
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

        packet_PRR = self.prr
        packet_PRR_down = packet_PRR / self.time_down_factor
        packet_DX_down = self.dx * self.space_down_factor  # m

        log.info(f"Sample rate from {packet_PRR}Hz to {packet_PRR_down}Hz")
        log.info(f"Dx from {self.dx}m to {packet_DX_down}m")
        del data

    def save_data(self):
        """
        Save the concatenated data to an HDF5 file and update metadata files.

        This method saves the concatenated data to an HDF5 file, creates necessary directories if they don't exist,
        updates metadata files such as 'last_start' and 'attrs.json', and resets the data_concat_offset.

        Returns:
            None
        """
        data_start_datetime = datetime.datetime.fromtimestamp(self.chunk_start_time)
        date_start_date_str = data_start_datetime.strftime("%Y%m%d")
        save_path = os.path.join(
            self.parent_output_dir,
            str(data_start_datetime.year),
            date_start_date_str + "_" + str(self.dir_index),
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        h5_concat = os.path.join(
            save_path,
            f"{date_start_date_str}_{str(self.chunk_start_time)}.h5",
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
            f.write(str(self.chunk_start_time) + "," + str(self.dir_index))
        # Save current dir to file
        self.save_last_dir()
        self.save_last_processed_file()

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

    def save_last_processed_file(self):
        """
        Save last processed file to file.

        Args:
            None

        Returns:
            None
        """

        # If last row of file is not the same as previous dir, save it
        try:
            with open(
                os.path.join(self.parent_output_dir, "processed_files"), "w"
            ) as f:
                f.write(self.df_file_data.iloc[self.index - 1]["files"])
        except IndexError:
            try:
                os.remove(os.path.join(self.parent_output_dir, "processed_files"))
            except FileNotFoundError:
                pass

    def read_last_processed_file(self):
        """
        Reads the last processed file from a file and returns it.

        Returns:
            The last processed file.

        """
        if os.path.exists(os.path.join(self.parent_output_dir, "processed_files")):
            with open(
                os.path.join(self.parent_output_dir, "processed_files"), "r"
            ) as f:
                processed_file = f.read()
                log.info("Last processed file:" + processed_file)
                return processed_file
        else:
            return None

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

    def read_processed_dirs(self) -> list[str]:
        """
        Reads the list of processed directories from a file and returns it.

        Returns:
            A list of processed directories.

        """
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

        if os.path.exists(os.path.join(self.parent_output_dir, ".last_dir")):
            with open(os.path.join(self.parent_output_dir, ".last_dir"), "r") as f:
                last_dir_str = f.read()
                if last_dir_str != self.current_dir:
                    # If last row of file is not the same as current dir, save it to persistent file
                    self.save_processed_dir(last_dir_str)
                    with open(
                        os.path.join(self.parent_output_dir, ".last_dir"), "w"
                    ) as f:
                        f.write(self.current_dir)
        else:
            with open(os.path.join(self.parent_output_dir, ".last_dir"), "w") as f:
                f.write(self.current_dir)

    def process_files(self):
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
            self.read_Prisma_segy()
            log.info(
                "read from segy chunk in %s seconds"
                % (datetime.datetime.now() - run_start_time).total_seconds()
            )
            self.save_data()  # save data to hdf5 file

            self.previous_dx = None
            self.previous_gauge_m = None
            self.previous_prr = None
            self.previous_dir = None

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
        processed_dirs = self.read_processed_dirs()
        dirs = [d for d in dirs if d not in processed_dirs]
        if "@Recycle" in dirs:
            dirs.remove("@Recycle")
        # if directory last modified less than 2 hours ago, skip it
        dirs = [
            d
            for d in dirs
            if (
                datetime.datetime.now()
                - datetime.datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(self.parent_input_dir, d))
                )
            ).total_seconds()
            >= 7200
        ]
        log.info("Skipped dirs modified less than 2 hours ago")

        self.df_file_data = pd.DataFrame(columns=["dirs", "files", "timestamps"])

        # Read last start time
        if os.path.exists(os.path.join(self.parent_output_dir, "last_start")):
            with open(os.path.join(self.parent_output_dir, "last_start"), "r") as f:
                try:
                    last_start_time, dir_index = f.read().split(",")
                    last_start_time = float(last_start_time)
                    self.data = h5py.File(
                        os.path.join(
                            self.parent_output_dir,
                            datetime.datetime.fromtimestamp(last_start_time).strftime(
                                "%Y/%Y%m%d" + "_" + str(dir_index)
                            ),
                            f"{datetime.datetime.fromtimestamp(last_start_time).strftime('%Y%m%d')}_{str(last_start_time)}.h5",
                        )
                    )["phase_down"][:]
                    self.data_concat = np.empty(
                        (
                            self.data.shape[0],
                            int(CHUNK_SIZE * PRR),
                        ),
                        dtype=np.dtype("f4"),
                    )
                    print(self.data_concat.shape)
                    self.data_concat[:, : self.data.shape[1]] = self.data
                    # Read previous JSON
                    with open(
                        os.path.join(
                            self.parent_output_dir,
                            datetime.datetime.fromtimestamp(last_start_time).strftime(
                                "%Y/%Y%m%d" + "_" + str(dir_index)
                            ),
                            "attrs.json",
                        )
                    ) as attrs_json:
                        attrs = json.load(attrs_json)
                        self.previous_dx = attrs["DX"]
                        self.previous_gauge_m = attrs["gauge_m"]
                        self.previous_prr = attrs["PRR"]

                    self.chunk_start_time = last_start_time
                    self.data_concat_offset = self.data.shape[1]
                    self.previous_channels = self.data_concat.shape[0]
                    self.dir_index = int(dir_index)
                    log.info("Resuming concatenation from last start time")
                    log.debug(f"Last start time: {last_start_time}")
                    log.debug(f"Shape: {self.data.shape}")
                except FileNotFoundError:
                    last_start_time = 0
                    self.dir_index = 0
        else:
            last_start_time = 0
            self.dir_index = 0

        for dir in dirs:
            files = os.listdir(os.path.join(self.parent_input_dir, dir))
            for file in files:
                if file.endswith(".segy"):
                    pass
                elif file.endswith(".json"):
                    files.remove(file)
                else:
                    log.warning(f"Unexpected file: {file}")
                    raise ValueError("Unexpected file in directory")
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

        # Remove already processed files
        last_processed_file = self.read_last_processed_file()
        if last_processed_file is not None:
            self.df_file_data = self.df_file_data[
                self.df_file_data["files"] > last_processed_file
            ]
        # Fix: last_start_time + offset
        self.time_diff = np.diff(
            np.concatenate(([last_start_time], self.df_file_data["timestamps"].values))
        )
        self.process_files()
        if self.current_dir is not None:
            self.save_processed_dir(self.current_dir)
        log.info("Concatenation finished")
