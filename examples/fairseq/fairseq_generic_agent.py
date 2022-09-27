
from simuleval.agents import SpeechToSpeechAgent
class FairseqSimulS2SAgent(SpeechToSpeechAgent):
    def __init__(self, args: Namespace, process_id: Optional[int] = None) -> None:
        super().__init__(args)
        self.source_segment_size = args.source_segment_size
        if process_id is None:
            process_id = 0
        assert process_id <= len(args.device) or args.device == ["cpu"]

        if args.device == ["cpu"]:
            self.set_device("cpu")
        else:
            self.set_device(args.device[process_id])

        logger.info(f"Loading target bpe tokenizer. (process id {os.getpid()})")
        self.load_target_bpe()
        logger.info(
            f"Loading model checkpoint and vocabulary. (process id {os.getpid()})"
        )
        self.load_model_vocab()
        logger.info(f"Loading TTS model. (process id {os.getpid()})")
        self.load_tts()

    def set_device(self, device: str) -> None:
        self.device = device
        try:
            torch.FloatTensor([1.0]).to(self.device)
            logger.info(f"Using device: {self.device}. (process id {os.getpid()})")
        except Exception as e:
            logger.error(f"Failed to use device: {self.device}, Error:")
            logger.error(f"Change to CPU")
            self.device = "cpu"

    def reset(self):
        super().reset()
        self.states["encoder_states"] = None
        self.states["source_samples"] = []
        self.states["target_indices"] = []
        self.states["target_subword_buffer"] = ""
        self.states["incremental_states"] = {}

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--checkpoint', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--fairseq-data", type=str, default=None,
                            help="Path of fairseq data binary")
        parser.add_argument("--fairseq-config", type=str, default=None,
                            help="Path to fairseq config yaml file")
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothsis if the source is not finished")
        parser.add_argument("--fixed-predicision-ratio", type=int, default=3,
                            help="Acoustic feature dimension.")
        parser.add_argument("--waitk-lagging", type=int, required=True,
                            help="Acoustic feature dimension.")
        parser.add_argument("--device", type=str, default="cuda:0", nargs="+",
                            help="Device to use")
        parser.add_argument("--source-segment-size", type=int, default=200,
                            help="Source segment size in ms")
        # fmt: on
        return parser

    def load_tts(self):
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False},
        )
        TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        self.tts_generator = task.build_generator(models, cfg)
        self.tts_task = task
        self.tts_models = [model.to(self.device) for model in models]

    def get_tts_output(self, text: str) -> Tuple[List[float], int]:
        sample = TTSHubInterface.get_model_input(self.tts_task, text)
        for key in sample["net_input"].keys():
            if sample["net_input"][key] is not None:
                sample["net_input"][key] = sample["net_input"][key].to(self.device)

            wav, rate = TTSHubInterface.get_prediction(
                self.tts_task, self.tts_models[0], self.tts_generator, sample
            )
            return wav, rate

    def load_target_bpe(self):
        # Get target tokenizer
        self.config = S2TDataConfig(
            Path(self.args.fairseq_data) / self.args.fairseq_config
        )
        self.target_bpe = build_bpe(Namespace(**self.config.bpe_tokenizer))

    def rename_state_dict(self, state) -> Dict:
        # Revolving offline parameter mismatch
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None
        state["cfg"]["model"]._name = TEST_ARCH_NAME
        state["cfg"]["model"].arch = TEST_ARCH_NAME
        state["cfg"]["model"].simul_type = "waitk_fixed_pre_decision"
        state["cfg"]["model"].noise_type = None
        state["cfg"]["model"].noise_mean = None
        state["cfg"]["model"].noise_var = None
        state["cfg"]["model"].energy_bias_init = 0
        state["cfg"]["model"].energy_bias = False
        state["cfg"]["model"].waitk_lagging = self.args.waitk_lagging
        state["cfg"]["model"].fixed_pre_decision_type = "average"
        state["cfg"][
            "model"
        ].fixed_pre_decision_ratio = self.args.fixed_predicision_ratio
        state["cfg"]["model"].fixed_pre_decision_pad_threshold = 0.3

        component_state_dict = OrderedDict()
        for key in state["model"].keys():
            if re.match(r"decoder\.layers\..\.encoder_attn", key):
                new_key = key.replace("k_proj", "k_proj_soft").replace(
                    "q_proj", "q_proj_soft"
                )
                component_state_dict[new_key] = state["model"][key]
                component_state_dict[key] = state["model"][key]
            else:
                component_state_dict[key] = state["model"][key]
        return component_state_dict

    def load_model_vocab(self):
        filename = self.args.checkpoint
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        task_args = state["cfg"]["task"]

        if self.args.fairseq_config is not None:
            task_args.config_yaml = self.args.fairseq_config

        if self.args.fairseq_data is not None:
            task_args.data = self.args.fairseq_data

        task = tasks.setup_task(task_args)

        component_state_dict = self.rename_state_dict(state)
        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(component_state_dict, strict=False)
        self.model.eval()
        self.model.share_memory()

        self.model.to(self.device)

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary

    def eval(self, **kwargs) -> None:
        with torch.no_grad():
            super().eval(**kwargs)

    def get_target_index_tensor(self) -> torch.LongTensor:
        return (
            torch.LongTensor(
                [self.model.decoder.dictionary.eos()] + self.states["target_indices"]
            )
            .to(self.device)
            .unsqueeze(0)
        )

    def set_steps(self, source: torch.FloatTensor, target: torch.LongTensor) -> None:
        self.states["incremental_states"]["steps"] = {
            "src": source.size(0),
            "tgt": target.size(1),
        }

    def get_next_target_full_word(self, force_decode: bool = False) -> Optional[str]:
        possible_full_words = self.target_bpe.decode(
            self.states["target_subword_buffer"]
        )
        if force_decode:
            # Return decoded full word anyways
            return possible_full_words if len(possible_full_words) > 0 else None

        possible_full_words_list = possible_full_words.split()
        if len(possible_full_words_list) > 1:
            self.states["target_subword_buffer"] = possible_full_words_list[1]
            return possible_full_words_list[0]

        # Not ready yet
        return None

    def process_read(self, source_info: Dict) -> Dict:
        self.states["source"].append(source_info)
        self.states["source_samples"] += source_info["segment"]
        self.is_finish_read = source_info["finished"]
        torch.cuda.empty_cache()
        self.states["encoder_states"] = self.model.encoder(
            torch.FloatTensor(self.states["source_samples"])
            .to(self.device)
            .unsqueeze(0),
            torch.LongTensor([len(self.states["source_samples"])]).to(self.device),
        )
        return source_info

    def process_write(self, prediction: str) -> str:
        self.states["target"].append(prediction)
        samples, fs = self.get_tts_output(prediction)
        samples = samples.cpu().tolist()
        return json.dumps({"samples": samples, "sample_rate": fs}).replace(" ", "")

    def update_target(self, pred_index):
        self.states["target_indices"].append(pred_index)
        bpe_token = self.model.decoder.dictionary.string([pred_index])

        if re.match(r"^\[.+_.+\]$", bpe_token):
            # Language Indicater
            return

        self.states["target_subword_buffer"] += " " + bpe_token

    def policy(self):
        # 0. Read at the beginning

        if self.states["encoder_states"] is None:
            self.read()
            return

        tgt_indices = self.get_target_index_tensor()
        self.set_steps(self.states["encoder_states"]["encoder_out"][0], tgt_indices)
        self.states["incremental_states"]["online"] = {
            "only": torch.tensor(not self.is_finish_read)
        }

        # 1.1 Run decoder
        decoder_states, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=self.states["encoder_states"],
            incremental_state=self.states["incremental_states"],
        )

        # 1.2. Make decision on model output
        if outputs.action == 0 and not self.is_finish_read:
            # 1.2.1 Read
            self.read()
        else:
            # 1.2.2 Predict next word
            lprobs = self.model.get_normalized_probs(
                [decoder_states[:, -1:]], log_probs=True
            )
            index = lprobs.argmax(dim=-1)[0, 0].item()
            # print(self.model.decoder.dictionary.string([index]))
            self.update_target(index)

            # 1.2.2 Only write full word to server
            is_finished = index == self.model.decoder.dictionary.eos()
            if is_finished:
                self.finish_eval()

            possible_full_word = self.get_next_target_full_word(
                force_decode=is_finished
            )

            if possible_full_word is None:
                self.read()
            else:
                self.write(possible_full_word)
