# coding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from pathlib import Path
from pydub import AudioSegment as pd 

import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode
import subprocess

from paddlespeech.cli.vector import VectorExecutor
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.utils import str2bool
from paddlespeech.vector.exps.ge2e.audio_processor import SpeakerVerificationPreprocessor
from paddlespeech.vector.models.lstm_speaker_encoder import LSTMSpeakerEncoder


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


def voice_cloning(args):
    
    am ='fastspeech2_aishell3'
    am_config = "D:\\workplaces\\PaddleModelData\\fastspeech2_aishell3_ckpt_vc2_1.2.0\\default.yaml"
    am_ckpt = 'D:\\workplaces\\PaddleModelData\\fastspeech2_aishell3_ckpt_vc2_1.2.0\\snapshot_iter_96400.pdz'
    am_stat = 'D:\\workplaces\\PaddleModelData\\fastspeech2_aishell3_ckpt_vc2_1.2.0\\speech_stats.npy'
    phones_dict = "D:\\workplaces\\PaddleModelData\\fastspeech2_aishell3_ckpt_vc2_1.2.0\\phone_id_map.txt"

    voc = 'pwgan_aishell3'
    voc_config = 'D:/workplaces/PaddleModelData/pwg_aishell3_ckpt_0.5/default.yaml'
    voc_ckpt = 'D:/workplaces/PaddleModelData/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz'
    voc_stat = 'D:/workplaces/PaddleModelData/pwg_aishell3_ckpt_0.5/feats_stats.npy'
    output_dir = "D:/workplaces/test/make/output"
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
    # input_dir = "D:\\workplaces\\PaddleModelData\\corpus"
    # input_dir = "D:\\wueryong\\onedrive\OneDrive - zut.edu.cn\\A_Doing\\corpus"
    audio_file_path = "D:\\workplaces\\test\\liyanqun\\test\\test.wav"

    # speaker encoder
    # if args.use_ecapa:
    vec_executor = VectorExecutor()
    # warm up
    print(">>>>>>>")
    print()
    
    # get the dir's first file 
    # audio_file=input_dir / os.listdir(input_dir)[0]

    
    audio_file_name = audio_file_path.split('\\')[-1].split(".")[0]
    audio_file_dir = audio_file_path.split('\\')[-2]
    
    audio_file_attr = pd.from_file(audio_file_path, format='wav')
    
    # audio_file_processed = Path(audio_file_path_processed)
    # audio_file_processed.mkdir(parents=True, exist_ok=True)
    audio_file_path_processed = "" 
    rate = audio_file_attr.frame_rate
    channels = audio_file_attr.channels
    flag = 0
    if rate != 16000 or channels !=1:
        audio_file_path_processed = audio_file_path.split(audio_file_name)[0] + "processed\\"
        audio_file_path_transfered = Path(audio_file_path_processed)
        audio_file_path_transfered.mkdir(parents=True, exist_ok=True)
        audio_file_new_path =  audio_file_path_processed+audio_file_name + ".wav"
        os.system("sox {} -r 16000 -b 16 -c 1 {}".format(audio_file_path , audio_file_new_path))
        audio_file_path = audio_file_new_path
        flag = 1 
    # audio_file_path = input_dir + audio_file.name
    # 复制一份wav文件保存audio_ok_name, 利用sox调整参数：通道-1 位-16 采样率-16k
    # subprocess.call(["sox {} -r 16000 -b 16 -c 1 {}".format(str(audio_file), str(audio_file))], shell=True) 
    # subprocess.call(["sox {} -r 16000 -b 16 -c 1 {}".format(input_dir.__str__()+"\\"+"record_20230225121113.wav", input_dir.__str__()+"\\"+"test.wav")], shell=True) 
    
    vec_executor(audio_file_path)

    # audio_file = Path(audio_file_path)

    print(">>>>>ECAPA-TDNN Done!")

    print("ECAPA-TDNN Done!")
    # use GE2E
    # else:
        # p = SpeakerVerificationPreprocessor(
        #     sampling_rate=16000,
        #     audio_norm_target_dBFS=-30,
        #     vad_window_length=30,
        #     vad_moving_average_width=8,
        #     vad_max_silence_length=6,
        #     mel_window_length=25,
        #     mel_window_step=10,
        #     n_mels=40,
        #     partial_n_frames=160,
        #     min_pad_coverage=0.75,
        #     partial_overlap_ratio=0.5)
        # print("Audio Processor Done!")

    #     speaker_encoder = LSTMSpeakerEncoder(
    #         n_mels=40, num_layers=3, hidden_size=256, output_size=256)
    #     speaker_encoder.set_state_dict(paddle.load(args.ge2e_params_path))
    #     speaker_encoder.eval()
    #     print("GE2E Done!")

    frontend = Frontend(phone_vocab_path=phones_dict)
    print("frontend done!")

    sentence = "今年天气怎么样？"
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
    # audio_file_name = audio_file_path_processed.split(".")[0]
    # ref_audio_path = input_dir / name
    # if args.use_ecapa:
    
    spk_emb = vec_executor(audio_file_path)

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

    sf.write(str(output_dir / (audio_file_name + ".wav")),
        wav.numpy(),
        samplerate=am_config.fs)
    print(f"{audio_file_name} done!")

    # # generate 5 random_spk_emb
    # for i in range(5):
    #     random_spk_emb = gen_random_embed(True)
    #     utt_id = "random_spk_emb"
    #     with paddle.no_grad():
    #         wav = voc_inference(am_inference(phone_ids, spk_emb=random_spk_emb))
    #     sf.write(
    #         str(output_dir / (utt_id + "_" + str(i) + ".wav")),
    #         wav.numpy(),
    #         samplerate=am_config.fs)
    # print(f"{utt_id} done!")


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
        default="澶╃尗绮剧伒",
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
    args = parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    voice_cloning(args)


if __name__ == "__main__":
    main()