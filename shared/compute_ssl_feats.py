#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
import torch
import sys
import logging
from kaldiio import ReadHelper, WriteHelper
import argparse
from transformers import AutoFeatureExtractor, AutoModel



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="SSL feature extraction for Kaldi",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "input",
    type=str,
    help="Input: scp:path/to/wav.scp or ark:- for stdin"
)
parser.add_argument(
    "output",
    type=str,
    help="Output: ark:path/to/output.ark or ark:- for stdout"
)
parser.add_argument(
    "-l", "--layer",
    type=int,
    default=12,
    help="SSL embedding layer to extract (e.g., 12)",
)
parser.add_argument(
    "--write-utt2dur",
    "-wud",
    type=str,
    default=None,
    help="Optional utt2dur output: ark,t:path/to/utt2dur or ark:path/to/utt2dur",
)
parser.add_argument(
    "--ssl-model",
    "-model",
    type=str,
    default="facebook/hubert-base-ls960",
    help="Pretrained SSL model type from HuggingFace",
)

args = parser.parse_args()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SSLExtractor:
    def __init__(self, model_id):
        """Loads the model and feature extractor exactly once into memory."""
        logger.info(f"Initializing model: {model_id}...")
        self.model_id = model_id
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        
        self.model.eval()
        self.model.encoder.layers = self.model.encoder.layers[:args.layer]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def extract(self, waveform, utt_id, target_layer=-1):
        
        try:
            # 1. Load audio at 16kHz
            #speech_array, sampling_rate = librosa.load(file_path, sr=16000)
            
            # 2. Pre-process into tensors
            inputs = self.extractor(waveform, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 3. Extract features without gradients
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            all_layers = outputs.hidden_states
            total_layers = len(all_layers)
            
            # Validate requested layer bounds
            if target_layer >= total_layers or target_layer < -total_layers:
                logger.error(f"Error for {utt_id}: Target layer {target_layer} out of bounds.")
                
                
            # 4. Extract target layer and store in CPU memory to save VRAM
            selected_features = all_layers[target_layer].cpu()
            
        except Exception as e:
            logger.error(f"Failed to process file {utt_id}: {str(e)}")
                
        return selected_features.squeeze(0).numpy()

def preprocess_waveform(waveform: np.ndarray) -> torch.Tensor:

    waveform = waveform.astype(np.float32)
    waveform /= np.iinfo(np.int16).max
    return torch.from_numpy(waveform).unsqueeze(0)


# ------------------------------------------------------------------------- #
# Main processing
# ------------------------------------------------------------------------- #
def process_features():
    logger.info("=" * 70)
    logger.info(f"Input:        {args.input}")
    logger.info(f"Output:       {args.output}")
    logger.info(f"Layer:        {args.layer}")
    logger.info("=" * 70)
    
    
    ssl_extractor = SSLExtractor(args.ssl_model)  

    utt2dur_data = {}
    processed = 0
    failed = 0

    if args.input == "ark:-":
        logger.info("Reading input from ark:- (stdin)")
        with ReadHelper("ark:-") as reader, WriteHelper(args.output) as writer:
            for utt_id, waveform in reader:
                try:
                    # waveform is assumed 1-D np.ndarray or similar
                    if isinstance(waveform, np.ndarray) and waveform.ndim == 1:
                        wf_t = torch.from_numpy(
                            waveform.astype(np.float32)
                        ).unsqueeze(0)
                    else:
                        wf_t = torch.tensor(
                            waveform, dtype=torch.float32
                        ).unsqueeze(0)
                    feats = ssl_extractor.extract(waveform.squeeze(), utt_id, target_layer=args.layer)


                    writer(utt_id, feats)
                    processed += 1
                except Exception as e:
                    logger.warning(f"Failed to process {utt_id}: {e}")
                    failed += 1
                    continue
    else:
        with ReadHelper(args.input) as reader, WriteHelper(args.output) as writer:
            for utt_id, (sample_rate, waveform) in reader:
                try:
                    duration = calculate_duration(waveform.shape[0], sample_rate)
                    utt2dur_data[utt_id] = duration

                    waveform = preprocess_waveform(waveform)
                    feats = ssl_extractor.extract(waveform.squeeze(), utt_id,  target_layer=args.layer, )

                    writer(utt_id, feats)
                    processed += 1

                    if processed % 100 == 0:
                        logger.info(f"Processed {processed} utterances...")
                except Exception as e:
                    logger.warning(f"Failed to process {utt_id}: {e}")
                    failed += 1
                    continue

    logger.info("=" * 70)
    logger.info(f"Successfully processed: {processed} utterances")
    if failed > 0:
        logger.warning(f"Failed: {failed} utterances")
    logger.info("=" * 70)

    return utt2dur_data


def write_utt2dur(utt2dur_data: dict):
    if not args.write_utt2dur:
        return
    out_spec = args.write_utt2dur
    if out_spec.startswith("ark,t:"):
        path = out_spec[6:]
    elif out_spec.startswith("ark:"):
        path = out_spec[4:]
    else:
        path = out_spec

    logger.info(f"Writing utt2dur to: {path}")
    try:
        with open(path, "w") as f:
            for utt_id, dur in sorted(utt2dur_data.items()):
                f.write(f"{utt_id} {dur:.3f}\n")
        logger.info(f"Wrote {len(utt2dur_data)} durations")
    except Exception as e:
        logger.error(f"Failed to write utt2dur: {e}")
        
        
def calculate_duration(num_samples: int, sample_rate: int) -> float:
    return float(num_samples) / float(sample_rate)


if __name__ == "__main__":
    try:
        utt2dur = process_features()
        write_utt2dur(utt2dur)
        logger.info("SSL feature extraction completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
              
