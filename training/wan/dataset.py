import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import resize


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]


def _normalize_exclude_videos_files(exclude_videos_file: Optional[Any]) -> List[str]:
    if not exclude_videos_file:
        return []
    if isinstance(exclude_videos_file, (str, Path)):
        return [str(exclude_videos_file)]
    try:
        return [str(path) for path in exclude_videos_file if path]
    except TypeError:
        return [str(exclude_videos_file)]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        use_gray_hand_videos: bool = False,
        prompt_subdir: str = "prompts",
        prompt_embeds_subdir: str = "prompt_embeds_rewrite_wan",
        static_video_subdir: str = "videos_static",
        hand_video_subdir: str = "videos_hands",
        hand_video_latents_subdir: str = "hand_video_latents_wan",
        video_latents_subdir: str = "video_latents_wan",
        static_video_latents_subdir: str = "static_video_latents_wan",
        image_latents_subdir: str = "image_latents_wan",
        fun_inp_i2v_latents_subdir: str = "fun_inp_i2v_latents_wan",
        clip_image_embeds_subdir: str = "clip_image_embeds",
        exclude_videos_file: Optional[Any] = None,
        align_width_to_32: bool = False,
        max_prompt_length: int = 512,
    ) -> None:
        super().__init__()
        self.align_width_to_32 = align_width_to_32
        self.max_prompt_length = max_prompt_length

        self.data_root = Path(data_root)
        # Handle dataset_file as list or single path
        if dataset_file is None:
            self.dataset_file_list: Optional[List[str]] = None
        elif isinstance(dataset_file, (list, tuple)):
            self.dataset_file_list = [str(path) for path in dataset_file]
        else:
            self.dataset_file_list = [str(dataset_file)]
        self.dataset_file = dataset_file
        self.caption_column = caption_column  # not used
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        if load_tensors:
            raise ValueError("DTWM release: load_tensors=True (pre-encoded latents) is not supported; "
                             "the trainer encodes raw videos on the fly.")
        self.random_flip = random_flip
        self.image_to_video = image_to_video
        self.prompt_subdir = prompt_subdir
        self.prompt_embeds_subdir = prompt_embeds_subdir
        self.static_video_subdir = static_video_subdir
        self.hand_video_subdir = hand_video_subdir
        self.hand_video_latents_subdir = hand_video_latents_subdir
        self.video_latents_subdir = video_latents_subdir
        self.static_video_latents_subdir = static_video_latents_subdir
        self.image_latents_subdir = image_latents_subdir
        self.fun_inp_i2v_latents_subdir = fun_inp_i2v_latents_subdir
        self.clip_image_embeds_subdir = clip_image_embeds_subdir
        self.exclude_videos_file = exclude_videos_file
        self.excluded_videos = self._load_excluded_video_entries(exclude_videos_file)
        
        # Ensure buckets are lists
        if not isinstance(self.height_buckets, list):
            self.height_buckets = [self.height_buckets]
        if not isinstance(self.width_buckets, list):
            self.width_buckets = [self.width_buckets]
        if not isinstance(self.frame_buckets, list):
            self.frame_buckets = [self.frame_buckets]
        
        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if self.dataset_file_list is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_datafile()
            # ) = self._load_dataset_from_csv()

        if self.excluded_videos:
            self.prompts, self.video_paths = self._apply_exclude_video_filter(self.prompts, self.video_paths)

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def _find_nearest_resolution(self, height: int, width: int) -> Tuple[int, int]:
        """Pick the nearest (H, W) resolution bucket, optionally aligning W to a multiple of 32.

        Note: alignment is only meaningful when a subclass actually resizes frames to the chosen bucket.
        """
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        nearest_h, nearest_w = nearest_res[1], nearest_res[2]

        # For WAN 2.2: align width to 32 for compatibility
        # 32 = 8 (VAE spatial_compression) * 4, ensures clean division through the pipeline
        if getattr(self, "align_width_to_32", False):
            # Always round to nearest multiple of 32, regardless of whether it's in buckets
            # This ensures WAN 2.2 compatibility even if the aligned value is not in the bucket list
            nearest_w = round(nearest_w / 32) * 32

        return nearest_h, nearest_w

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_excluded_video_entries(self, exclude_videos_file: Optional[Any]) -> set[str]:
        exclude_paths_raw = _normalize_exclude_videos_files(exclude_videos_file)
        if not exclude_paths_raw:
            return set()

        entries: set[str] = set()
        for exclude_path_raw in exclude_paths_raw:
            exclude_path = Path(exclude_path_raw)
            if not exclude_path.is_absolute():
                exclude_path = self.data_root / exclude_path
            if not exclude_path.exists():
                raise FileNotFoundError(f"Exclude videos file not found: {exclude_path}")

            path_entries = {
                line.strip()
                for line in exclude_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            entries.update(path_entries)
            logger.info("Loaded %d excluded videos from %s", len(path_entries), exclude_path)

        logger.info("Loaded %d unique excluded videos from %d file(s)", len(entries), len(exclude_paths_raw))
        return entries

    def _video_path_to_rel_key(self, path: Path) -> str:
        try:
            return path.relative_to(self.data_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _get_candidate_action_dirs(self, path: Path) -> List[Path]:
        action_dir = path.parent.parent
        candidate_dirs = [action_dir]
        if action_dir.name == "processed2":
            candidate_dirs.append(action_dir.parent)
        return candidate_dirs

    def _resolve_existing_subdir(self, path: Path, subdir: str) -> Path:
        candidate_dirs = [action_dir / subdir for action_dir in self._get_candidate_action_dirs(path)]
        for candidate_dir in candidate_dirs:
            if candidate_dir.exists():
                return candidate_dir
        return candidate_dirs[0]

    def _resolve_existing_file(self, path: Path, subdir: str, filename: str) -> Path:
        candidate_files = [
            action_dir / subdir / filename for action_dir in self._get_candidate_action_dirs(path)
        ]
        for candidate_file in candidate_files:
            if candidate_file.exists():
                return candidate_file
        return candidate_files[0]

    def _apply_exclude_video_filter(
        self, prompts: List[str], video_paths: List[Path]
    ) -> Tuple[List[str], List[Path]]:
        filtered_prompts: List[str] = []
        filtered_video_paths: List[Path] = []
        excluded_count = 0

        for prompt, video_path in zip(prompts, video_paths):
            rel_key = self._video_path_to_rel_key(video_path)
            if rel_key in self.excluded_videos:
                excluded_count += 1
                continue
            filtered_prompts.append(prompt)
            filtered_video_paths.append(video_path)

        if excluded_count:
            logger.warning(
                "Excluded %d videos listed in %s from the dataset.",
                excluded_count,
                self.exclude_videos_file,
            )
        if not filtered_video_paths:
            raise ValueError("All dataset videos were excluded. Check exclude_videos_file.")

        return filtered_prompts, filtered_video_paths

    def _build_raw_clip_plan(self, path: Path) -> Dict[str, int]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        return self._build_raw_clip_plan_from_reader(video_reader)

    def _build_raw_clip_plan_from_reader(self, video_reader) -> Dict[str, int]:
        video_num_frames = len(video_reader)
        if video_num_frames <= 0:
            raise ValueError("Video has no frames")

        first_frame = video_reader[0]
        height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
        target_height, target_width = self._find_nearest_resolution(height, width)
        target_num_frames = int(self.max_num_frames)
        max_start = max(video_num_frames - target_num_frames, 0)
        start_idx = random.randint(0, max_start) if max_start > 0 else 0

        return {
            "start_idx": start_idx,
            "target_num_frames": target_num_frames,
            "target_height": target_height,
            "target_width": target_width,
        }

    def _resolve_raw_clip_plan(self, video_reader, clip_plan: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        if clip_plan is None:
            return self._build_raw_clip_plan_from_reader(video_reader)

        resolved_plan = dict(clip_plan)
        target_num_frames = int(resolved_plan.get("target_num_frames", self.max_num_frames))
        video_num_frames = len(video_reader)
        max_start = max(video_num_frames - target_num_frames, 0)
        resolved_plan["target_num_frames"] = target_num_frames
        resolved_plan["start_idx"] = min(int(resolved_plan.get("start_idx", 0)), max_start)
        return resolved_plan

    def _load_raw_frames(self, video_reader, clip_plan: Dict[str, int]) -> torch.Tensor:
        video_num_frames = len(video_reader)
        target_num_frames = int(clip_plan["target_num_frames"])
        start_idx = int(clip_plan["start_idx"])
        end_idx = min(video_num_frames, start_idx + target_num_frames)
        frame_indices = list(range(start_idx, end_idx))

        if not frame_indices:
            raise ValueError("No frame indices selected for video clip")

        frames = video_reader.get_batch(frame_indices).float()
        frames = frames.permute(0, 3, 1, 2).contiguous()

        if frames.shape[0] < target_num_frames:
            pad_frames = frames[-1:].repeat(target_num_frames - frames.shape[0], 1, 1, 1)
            frames = torch.cat([frames, pad_frames], dim=0)

        return frames

    def _get_raw_video_sample(self, index: int) -> Tuple[Dict[str, Any], Dict[str, int]]:
        clip_plan = self._build_raw_clip_plan(self.video_paths[index])
        image, video, _ = self._preprocess_video(self.video_paths[index], clip_plan=clip_plan)

        sample = {
            "prompt": self.id_token + self.prompts[index],
            "image": image,
            "video": video,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
        }
        return sample, clip_plan

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # if isinstance(index, list):
        #     # Here, index is actually a list of data objects that we need to return.
        #     # The BucketSampler should ideally return indices. But, in the sampler, we'd like
        #     # to have information about num_frames, height and width. Since this is not stored
        #     # as metadata, we need to read the video to get this information. You could read this
        #     # information without loading the full video in memory, but we do it anyway. In order
        #     # to not load the video twice (once to get the metadata, and once to return the loaded video
        #     # based on sampled indices), we cache it in the BucketSampler. When the sampler is
        #     # to yield, we yield the cache data instead of indices. So, this special check ensures
        #     # that data is not loaded a second time. PRs are welcome for improvements.
        #     return index

        # Handle index out of range with mod operation to prevent training interruption
        original_index = index
        if index >= len(self.video_paths):
            logger.warning(f"Index {index} is out of range (dataset size: {len(self.video_paths)}). Using mod operation: {index} % {len(self.video_paths)} = {index % len(self.video_paths)}")
            index = index % len(self.video_paths)

        sample, _ = self._get_raw_video_sample(index)
        return sample

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        video_path = self.data_root.joinpath(self.video_column)

        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        # Derive prompt paths from video paths (same as _load_dataset_from_datafile)
        prompt_paths = [path.parent.parent.joinpath("prompts", path.name.replace(".mp4", ".txt")) for path in video_paths]
        # prompts = [path.read_text().strip() for path in prompt_paths]
        prompts = ["" for path in prompt_paths]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_prompt_texts(self, prompt_paths: List[Path], dataset_path: Path) -> List[str]:
        prompts: List[str] = []
        missing_count = 0
        for prompt_path in prompt_paths:
            if prompt_path.is_file():
                prompts.append(prompt_path.read_text(encoding="utf-8"))
            else:
                prompts.append("")
                missing_count += 1
        if missing_count:
            logger.warning(
                "Prompt files missing for %d samples while loading %s. Using empty prompts.",
                missing_count,
                dataset_path,
            )
        return prompts

    def _load_dataset_from_datafile(self) -> Tuple[List[str], List[Path]]:
        """Load dataset from one or more datafiles (supports list of files)."""
        if not self.dataset_file_list:
            raise ValueError("dataset_file should not be empty when provided.")
        
        all_prompts: List[str] = []
        all_video_paths: List[Path] = []
        
        for dataset_file in self.dataset_file_list:
            dataset_path = self.data_root / dataset_file
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            with open(dataset_path, "r", encoding="utf-8") as f:
                file_video_paths = [
                    self.data_root.joinpath(line.strip()) for line in f.readlines() if len(line.strip()) > 0
                ]
            
            prompt_paths = [
                self._resolve_existing_file(path, self.prompt_subdir, path.name.replace(".mp4", ".txt"))
                for path in file_video_paths
            ]
            file_prompts = self._load_prompt_texts(prompt_paths, dataset_path)

            all_video_paths.extend(file_video_paths)
            all_prompts.extend(file_prompts)
        
        return all_prompts, all_video_paths

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single raw video and returns (image, video, prompt_embed=None).

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.
        """
        try:
            video_reader = decord.VideoReader(uri=path.as_posix())
            resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
            frames = self._load_raw_frames(video_reader, resolved_clip_plan)
            frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None
        except Exception as e:
            logger.error(f"Failed to preprocess video {path}: {e}")
            raise e


class VideoDatasetWithConditions(VideoDataset):
    """
    Extended VideoDataset that supports additional condition videos:
    - hand_videos: Egocentric hand mesh videos (automatically derived from main video paths)
    - static_videos: Static scene videos (automatically derived from main video paths)
    """

    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        use_gray_hand_videos: bool = False,
        prompt_subdir: str = "prompts",
        prompt_embeds_subdir: str = "prompt_embeds_rewrite_wan",
        static_video_subdir: str = "videos_static",
        static_video_source_mode: str = "directory",
        hand_video_subdir: str = "videos_hands",
        hand_video_future_subdir: Optional[str] = None,
        hand_video_latents_subdir: str = "hand_video_latents_wan",
        static_disparity_subdir: Optional[str] = None,
        static_disparity_latents_subdir: Optional[str] = None,
        hand_disparity_subdir: Optional[str] = None,
        hand_disparity_latents_subdir: Optional[str] = None,
        video_latents_subdir: str = "video_latents_wan",
        static_video_latents_subdir: str = "static_video_latents_wan",
        image_latents_subdir: str = "image_latents_wan",
        fun_inp_i2v_latents_subdir: str = "fun_inp_i2v_latents_wan",
        clip_image_embeds_subdir: str = "clip_image_embeds",
        exclude_videos_file: Optional[Any] = None,
        require_static_videos: bool = True,
        align_width_to_32: bool = False,
        max_prompt_length: int = 512,
    ) -> None:
        # Initialize parent class with main video column
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
            max_prompt_length=max_prompt_length,
            prompt_subdir=prompt_subdir,
            prompt_embeds_subdir=prompt_embeds_subdir,
            static_video_subdir=static_video_subdir,
            hand_video_subdir=hand_video_subdir,
            hand_video_latents_subdir=hand_video_latents_subdir,
            video_latents_subdir=video_latents_subdir,
            static_video_latents_subdir=static_video_latents_subdir,
            image_latents_subdir=image_latents_subdir,
            fun_inp_i2v_latents_subdir=fun_inp_i2v_latents_subdir,
            clip_image_embeds_subdir=clip_image_embeds_subdir,
            exclude_videos_file=exclude_videos_file,
            align_width_to_32=align_width_to_32,
        )

        # Store the use_gray_hand_videos flag
        self.use_gray_hand_videos = use_gray_hand_videos
        self.static_disparity_subdir = static_disparity_subdir
        self.static_disparity_latents_subdir = static_disparity_latents_subdir
        self.hand_disparity_subdir = hand_disparity_subdir
        self.hand_disparity_latents_subdir = hand_disparity_latents_subdir
        self.static_video_source_mode = str(static_video_source_mode).strip().lower()
        self.require_static_videos = bool(require_static_videos)
        if self.static_video_source_mode not in {"directory", "copy_first_frame"}:
            raise ValueError(
                f"Unsupported static_video_source_mode: {static_video_source_mode}. "
                "Use one of: directory, copy_first_frame."
            )
        # Store the align_width_to_32 flag (for WAN 2.2 compatibility)
        self.align_width_to_32 = align_width_to_32
        
        # Automatically derive hand video and static video paths from main video paths
        if self.use_gray_hand_videos:
            self.hand_video_paths = self._derive_condition_video_paths("videos_hands_gray")
        else:
            self.hand_video_paths = (
                self._derive_condition_video_paths(self.hand_video_subdir)
                if self.hand_video_subdir
                else None
            )
        # DTWM AR: optional SECOND hand-conditioning stream (data.hand_video_future_subdir). The train
        # loop splices [observed skeleton 0..K | future stream K+1..K+ar_action_horizon | black] per prefix K.
        self.hand_video_future_subdir = hand_video_future_subdir
        self.hand_video_future_paths = (
            self._derive_condition_video_paths(hand_video_future_subdir)
            if hand_video_future_subdir
            else None
        )
        self.static_video_paths = self._derive_condition_video_paths(self.static_video_subdir)
        self.static_disparity_paths = (
            self._derive_condition_video_paths(self.static_disparity_subdir)
            if self.static_disparity_subdir
            else None
        )
        self.hand_disparity_paths = (
            self._derive_condition_video_paths(self.hand_disparity_subdir)
            if self.hand_disparity_subdir
            else None
        )
        # Validate that all condition videos exist
        if not self.load_tensors:
            if self.hand_video_paths is not None and any(not path.is_file() for path in self.hand_video_paths):
                missing_hand_videos = [path for path in self.hand_video_paths if not path.is_file()]
                raise ValueError(
                    f"Some hand video files are missing. First few missing files: {missing_hand_videos[:5]}"
                )
            
            # if any(not path.is_file() for path in self.hand_video_gray_paths):
            #     missing_hand_gray_videos = [path for path in self.hand_video_gray_paths if not path.is_file()]
            #     raise ValueError(
            #         f"Some hand gray video files are missing. First few missing files: {missing_hand_gray_videos[:5]}"
            #     )
            
            if self.require_static_videos and self.static_video_source_mode == "directory":
                missing_static_videos = [path for path in self.static_video_paths if not path.is_file()]
                if missing_static_videos:
                    raise ValueError(
                        f"Some static video files are missing. First few missing files: {missing_static_videos[:5]}"
                    )
            if self.static_disparity_paths is not None and any(
                not path.is_file() for path in self.static_disparity_paths
            ):
                missing_static_disparities = [path for path in self.static_disparity_paths if not path.is_file()]
                raise ValueError(
                    "Some static disparity video files are missing. "
                    f"First few missing files: {missing_static_disparities[:5]}"
                )
            if self.hand_disparity_paths is not None and any(
                not path.is_file() for path in self.hand_disparity_paths
            ):
                missing_hand_disparities = [path for path in self.hand_disparity_paths if not path.is_file()]
                raise ValueError(
                    "Some hand disparity video files are missing. "
                    f"First few missing files: {missing_hand_disparities[:5]}"
                )

    def _derive_condition_video_paths(self, condition_folder: str) -> List[Path]:
        """Derive condition video paths from main video paths.
        
        Args:
            condition_folder: Name of the condition folder (e.g., "videos_hands", "videos_static")
            
        Returns:
            List of condition video paths
        """
        condition_paths = []
        
        for video_path in self.video_paths:
            # Example: video_path = /path/to/sequences/videos/00001.mp4
            # We want: /path/to/sequences/videos_hands/00001.mp4
            # or: /path/to/sequences/videos_static/00001.mp4
            
            parent_dir = video_path.parent
            filename = video_path.name
            condition_path = parent_dir.parent / condition_folder / filename
            if not condition_path.exists() and parent_dir.parent.name == "processed2":
                fallback_condition_path = parent_dir.parent.parent / condition_folder / filename
                if fallback_condition_path.exists():
                    condition_path = fallback_condition_path
            condition_paths.append(condition_path)
        
        return condition_paths

    def _get_item_once(self, index: int) -> Dict[str, Any]:
        main_data, clip_plan = self._get_raw_video_sample(index)

        # Load raw videos for condition videos
        try:
            if self.hand_video_paths is None:
                main_data["hand_videos"] = None
            else:
                _, hand_video, _ = self._preprocess_video(self.hand_video_paths[index], clip_plan=clip_plan)
                main_data["hand_videos"] = hand_video
        except Exception as e:
            logger.warning(f"Failed to load hand video for index {index}: {e}")
            main_data["hand_videos"] = None

        # DTWM AR: second hand stream (data.hand_video_future_subdir), SAME clip_plan alignment. The trainer
        # splices [observed skeleton 0..K | future stream K+1..K+ar_action_horizon | black] per prefix K.
        try:
            if self.hand_video_future_paths is None:
                main_data["hand_videos_future"] = None
            else:
                _, hvf, _ = self._preprocess_video(self.hand_video_future_paths[index], clip_plan=clip_plan)
                main_data["hand_videos_future"] = hvf
        except Exception as e:
            logger.warning(f"Failed to load future hand video for index {index}: {e}")
            main_data["hand_videos_future"] = None

        # DTWM: per-clip tactile pressure grid <action_dir>/pressure/<stem>.npy, (L, 2, 21, 21), frame-aligned
        # via the SAME clip_plan as the video. Optional; clips without it train without tactile input.
        try:
            vp = self.video_paths[index]
            pp = vp.parent.parent / "pressure" / (vp.stem + ".npy")
            if pp.is_file():
                P = np.load(pp).astype("float32")            # (Lfull, 2, 21, 21)
                s = int(clip_plan["start_idx"]); nf = int(clip_plan["target_num_frames"])
                P = P[s:s + nf]
                if P.shape[0] < nf:                          # mirror video last-frame padding
                    P = np.concatenate([P, np.repeat(P[-1:], nf - P.shape[0], axis=0)], axis=0)
                main_data["pressure"] = torch.from_numpy(P)  # (nf, 2, 21, 21) float32
            else:
                main_data["pressure"] = None
        except Exception as e:
            logger.warning(f"Failed to load pressure for index {index}: {e}")
            main_data["pressure"] = None

        # DTWM: per-clip per-hand centroids <action_dir>/hand_centroids/<stem>.npy, (L, 2, 3) =
        # (y_norm, x_norm, valid), for the spatial tactile injection and the hand-region loss weight,
        # frame-aligned via the SAME clip_plan as the video/pressure. Optional.
        try:
            vp = self.video_paths[index]
            cp = vp.parent.parent / "hand_centroids" / (vp.stem + ".npy")
            if cp.is_file():
                C = np.load(cp).astype("float32")            # (Lfull_or_clip, 2, 3)
                s = int(clip_plan["start_idx"]); nf = int(clip_plan["target_num_frames"])
                C = C[s:s + nf]
                if C.shape[0] < nf:
                    C = np.concatenate([C, np.repeat(C[-1:], nf - C.shape[0], axis=0)], axis=0)
                main_data["hand_centroids"] = torch.from_numpy(C)  # (nf, 2, 3)
            else:
                main_data["hand_centroids"] = None
        except Exception as e:
            logger.warning(f"Failed to load hand_centroids for index {index}: {e}")
            main_data["hand_centroids"] = None

        if not self.require_static_videos:
            main_data["static_videos"] = None
        elif self.static_video_source_mode == "copy_first_frame":
            main_video = main_data["video"]
            main_data["static_videos"] = main_video[0:1].repeat(main_video.shape[0], 1, 1, 1).clone()
        else:
            try:
                _, static_video, _ = self._preprocess_video(self.static_video_paths[index], clip_plan=clip_plan)
                main_data["static_videos"] = static_video
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load static video for index {index}: {self.static_video_paths[index]}"
                ) from e

        if self.static_disparity_paths is not None:
            try:
                _, static_disparity, _ = self._preprocess_video(
                    self.static_disparity_paths[index], clip_plan=clip_plan
                )
                main_data["static_disparities"] = static_disparity
            except Exception as e:
                logger.warning(f"Failed to load static disparity video for index {index}: {e}")
                main_data["static_disparities"] = None
        else:
            main_data["static_disparities"] = None

        if self.hand_disparity_paths is not None:
            try:
                _, hand_disparity, _ = self._preprocess_video(
                    self.hand_disparity_paths[index], clip_plan=clip_plan
                )
                main_data["hand_disparities"] = hand_disparity
            except Exception as e:
                logger.warning(f"Failed to load hand disparity video for index {index}: {e}")
                main_data["hand_disparities"] = None
        else:
            main_data["hand_disparities"] = None

        return main_data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # if isinstance(index, list):
        #     return index

        if index >= len(self.video_paths):
            logger.warning(
                "Index %d is out of range (dataset size: %d). Using mod operation: %d %% %d = %d",
                index,
                len(self.video_paths),
                index,
                len(self.video_paths),
                index % len(self.video_paths),
            )
            index = index % len(self.video_paths)

        return self._get_item_once(index)


class VideoDatasetWithConditionsAndResizing(VideoDatasetWithConditions):
    """
    Extended VideoDatasetWithConditions that also supports resizing.
    """
    
    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution
