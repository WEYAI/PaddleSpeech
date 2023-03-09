# coding=utf-8
import argparse
import os
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf
import yaml
import configparser
from yacs.config import CfgNode

from paddlespeech.cli.vector import VectorExecutor
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.utils import str2bool
from paddlespeech.vector.exps.ge2e.audio_processor import SpeakerVerificationPreprocessor
from paddlespeech.vector.models.lstm_speaker_encoder import LSTMSpeakerEncoder


#实例化configParser对象
config = configparser.ConfigParser()
path = os.path.abspath(".") + '\\voice_cloning.ini'
#read读取ini文件,设定编解码方式
#config.read('voice_cloning.ini', encoding='utf-8')
config.read(path)

input_dir = config['P']['input_dir']
print(input_dir)
output_dir = config['P']['output_dir']
print(output_dir)
sentence = config['P']['sentence']
print(sentence)



def gen_random_embed(use_ecapa: bool=False):
    if use_ecapa:
        # Randomly generate numbers of -25 ~ 25, 192 is the dim of spk_emb
        random_spk_emb = (-1 + 2 * np.random.rand(192)) * 25

    # GE2E
    else:
        # Randomly generate numbers of 0 ~ 0.2, 256 is the dim of spk_emb
        random_spk_emb = np.random.rand(256) * 0.2
    random_spk_emb = paddle.to_tensor(random_spk_emb, dtype='float32')
    return random_spk_emb


def voice_cloning(input_dir,output_dir,sentence):
    args = parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    am ='fastspeech2_aishell3'
    am_config = "./PaddleModelData/fastspeech2_aishell3_ckpt_vc2_1.2.0/default.yaml"
    am_ckpt = './PaddleModelData/fastspeech2_aishell3_ckpt_vc2_1.2.0/snapshot_iter_96400.pdz'
    am_stat = './PaddleModelData/fastspeech2_aishell3_ckpt_vc2_1.2.0/speech_stats.npy'
    phones_dict = "./PaddleModelData/fastspeech2_aishell3_ckpt_vc2_1.2.0/phone_id_map.txt"

    voc = 'pwgan_aishell3'
    voc_config = './PaddleModelData/pwg_aishell3_ckpt_0.5/default.yaml'
    voc_ckpt = './PaddleModelData/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz'
    voc_stat = './PaddleModelData/pwg_aishell3_ckpt_0.5/feats_stats.npy'
    ge2e_params_path = ""

    # Init body.
    with open(am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))
    with open(voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(am_config)
    print(voc_config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = Path(input_dir)

    # speaker encoder
    # if args.use_ecapa:
    vec_executor = VectorExecutor()
    # warm up
    vec_executor(audio_file=input_dir / os.listdir(input_dir)[0])
    print(">>>>>ECAPA-TDNN Done!")

    print("ECAPA-TDNN Done!")

    frontend = Frontend(phone_vocab_path=phones_dict)
    print("frontend done!")

    
    input_ids = frontend.get_input_ids(sentence, merge_sentences=True)
    phone_ids = input_ids["phone_ids"][0]

    # acoustic model
    am_inference = get_am_inference(
        am=am,
        am_config=am_config,
        am_ckpt=am_ckpt,
        am_stat=am_stat,
        phones_dict=phones_dict)

    # vocoder
    voc_inference = get_voc_inference(
        voc=voc,
        voc_config=voc_config,
        voc_ckpt=voc_ckpt,
        voc_stat=voc_stat)

    for name in os.listdir(input_dir):
        utt_id = name.split(".")[0]
        ref_audio_path = input_dir / name
        # if args.use_ecapa:
        spk_emb = vec_executor(audio_file=ref_audio_path)
        spk_emb = paddle.to_tensor(spk_emb)
        # GE2E
        # else:
        #     mel_sequences = p.extract_mel_partials(
        #         p.preprocess_wav(ref_audio_path))
        #     with paddle.no_grad():
        #         spk_emb = speaker_encoder.embed_utterance(
        #             paddle.to_tensor(mel_sequences))
        with paddle.no_grad():
            wav = voc_inference(am_inference(phone_ids, spk_emb=spk_emb))

        sf.write(
            str(output_dir / (utt_id + ".wav")),
            wav.numpy(),
            samplerate=am_config.fs)
        print(f"{utt_id} done!")

    # generate 5 random_spk_emb
    '''
    for i in range(5):
        random_spk_emb = gen_random_embed(True)
        utt_id = "random_spk_emb"
        with paddle.no_grad():
            wav = voc_inference(am_inference(phone_ids, spk_emb=random_spk_emb))
            output_file = str(output_dir / (utt_id + "_" + str(i) + ".wav"))
        sf.write(
            output_file,
            wav.numpy(),
            samplerate=am_config.fs)
    '''
    print(f"{utt_id} done!")
    
    return output_file


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--am',
        type=str,
        default='fastspeech2_csmsc',
        choices=['fastspeech2_aishell3', 'tacotron2_aishell3'],
        help='Choose acoustic model type of tts task.')
    parser.add_argument(
        '--am_config', type=str, default="", help='Config of acoustic model.')
    parser.add_argument(
        '--am_ckpt',
        type=str,
        default=None,
        help='Checkpoint file of acoustic model.')
    parser.add_argument(
        "--am_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training acoustic model."
    )
    parser.add_argument(
        "--phones-dict",
        type=str,
        default="phone_id_map.txt",
        help="phone vocabulary file.")
    # vocoder
    parser.add_argument(
        '--voc',
        type=str,
        default='pwgan_csmsc',
        choices=['pwgan_aishell3'],
        help='Choose vocoder type of tts task.')

    parser.add_argument(
        '--voc_config', type=str, default=None, help='Config of voc.')
    parser.add_argument(
        '--voc_ckpt', type=str, default=None, help='Checkpoint file of voc.')
    parser.add_argument(
        "--voc_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training voc."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="天猫精灵",
        help="text to synthesize, a line")
    parser.add_argument(
        "--ge2e_params_path", type=str, help="ge2e params path.")
    parser.add_argument(
        "--use_ecapa",
        type=str2bool,
        default=False,
        help="whether to use ECAPA-TDNN as speaker encoder.")
    parser.add_argument(
        "--ngpu", type=int, default=0, help="if ngpu=0, use cpu.")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="input dir of *.wav, the sample rate will be resample to 16k.")
    parser.add_argument("--output-dir", type=str, help="output dir.")

    args = parser.parse_args()
    return args


def main():
    
    output_dir = "./output"
    input_dir = "./input"
    sentence = "我在天猫网站上买了一个天猫精灵，我使用天猫精灵听天气预报！"
    voice_cloning(input_dir,output_dir,sentence)


if __name__ == "__main__":
    print(voice_cloning(input_dir,output_dir,sentence))
