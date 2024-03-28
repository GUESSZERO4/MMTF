import os
import logging
import torch
import sys
import time
import json
import copy
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from spacy.lang.en import English
from collections import Counter
from pathlib import Path


nlp = English()
logger = logging.getLogger(__name__)

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

def msareadfile(filename):
    data = pd.read_csv(filename, sep=',')
    img_ids = (np.array(data['id'].values, dtype=object)).tolist()
    labels = np.array(pd.read_csv(filename, encoding='gbk', sep=',', usecols=['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']), dtype=object)
    print("The number of samples: " + str(len(labels)))
    return img_ids, labels



def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_mmtxt(cls, input_file, quotechar=None):
        return msareadfile(input_file)


class MSAInputExample(object):
    def __init__(self, img_id, label):
        self.label = label.tolist()
        self.img_id = img_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class MNERInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids, img_feat, img_ti_feat, caption,
                 caption_len):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_len = input_len
        self.segment_ids = segment_ids
        self.img_feat = img_feat
        self.label_ids = label_ids
        self.img_ti_feat = img_ti_feat
        self.caption = caption
        self.caption_len = caption_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class MSAProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        img_ids, labels = self._read_mmtxt(
            os.path.join(data_dir, "train.csv"))
        return self._create_examples(img_ids, labels)

    def get_dev_examples(self, data_dir):
        """See base class."""
        img_ids, labels = self._read_mmtxt(
            os.path.join(data_dir, "valid.csv"))
        return self._create_examples(img_ids, labels)

    def get_test_examples(self, data_dir):
        """See base class."""
        img_ids, labels = self._read_mmtxt(
            os.path.join(data_dir, "test.csv"))
        return self._create_examples(img_ids, labels)

    def get_labels(self):
        # return ["positive", "neutral", "negative"]
        return ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    def _create_examples(self, img_ids, labels):
        examples = []
        for i in range(len(labels)):
            examples.append(MSAInputExample(img_id=img_ids[i], label=labels[i]))
        return examples


def convert_one_msa_example_to_feature(example, label_list, tokenizer, max_seq_length, path_img, vocabulary, cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                       sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0, sequence_a_segment_id=0, mask_padding_with_zero=True,
                                       crop_size=224, ti_crop_size=32):
    # label_map = {label: i for i, label in enumerate(label_list)}  # dict
    # print("label:", example.label)
    # label = label_map[example.label]
    label = example.label

    transform = transforms.Compose([
        transforms.Resize([crop_size, crop_size]),  # 调整图片到指定的大小
        transforms.ToTensor(),
        transforms.Normalize((0.48, 0.498, 0.531),
                             (0.214, 0.207, 0.207))])

    transform_for_ti = transforms.Compose([
        transforms.Resize([ti_crop_size, ti_crop_size]),  # 调整图片到指定的大小
        transforms.ToTensor(),
        transforms.Normalize((0.48, 0.498, 0.531),
                             (0.214, 0.207, 0.207))])

    text_name = path_img + '/' + str(example.img_id) + '.txt'
    with open(text_name, 'r', encoding='unicode_escape') as f:
        text = f.readlines()

    tokens = tokenizer.tokenize(text[0])

    # Account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]

    tokens += [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    input_len = len(input_ids)
    # Zero-pad up to the sequence length.
    if len(input_ids) < max_seq_length:
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] *
                           padding_length) + segment_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length

    image_name = str(example.img_id) + '.png'
    image_path = os.path.join(path_img, image_name)
    if not os.path.exists(image_path):
        print('Not exist:', image_path)
    try:
        # print('Image path', image_path)
        image_feat = image_process(image_path, transform)
        image_ti_feat = image_process(image_path, transform_for_ti)
    except:
        image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
        image_feat = image_process(image_path_fail, transform)
        image_ti_feat = image_process(image_path_fail, transform_for_ti)

    caption = []
    for word in vocabulary.tokenizer_eng(text[0]):
        caption += vocabulary.numericalize(word)
    if len(caption) > max_seq_length:
        caption = caption[0:(max_seq_length - 2)]
    caption_len = [len(caption) + 2]
    caption = [vocabulary.stoi["<SOS>"]] + caption + [vocabulary.stoi["<EOS>"]]
    while len(caption) < max_seq_length:
        caption.append(vocabulary.stoi["<PAD>"])

    # print('label_ids', len(label_ids))
    # print(label_ids)

    # print('input_ids', input_ids)
    # print('input_mask', input_mask)
    # print('input_len', input_len)
    # print('segment_ids', segment_ids)
    # print('label', label)
    # print('img_feat', image_feat)
    # print('img_ti_feat', image_ti_feat)
    # print('caption', caption)
    # print('caption_len', caption_len)
    return MNERInputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len, segment_ids=segment_ids,
                             label_ids=label, img_feat=image_feat, img_ti_feat=image_ti_feat, caption=caption, caption_len=caption_len)



class MSADataset(Dataset):
    def __init__(self, examples, label_list, max_seq_length, tokenizer, crop_size, path_img, ti_crop_size, vocabulary, use_xlmr=False):
        self.examples = examples
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.crop_size = crop_size
        self.path_img = path_img
        self.ti_crop_size = ti_crop_size
        self.vocabulary = vocabulary
        self.use_xlmr = use_xlmr

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        if not self.use_xlmr:
            feature = convert_one_msa_example_to_feature(example=example, label_list=self.label_list, tokenizer=self.tokenizer,
                                                         max_seq_length=self.max_seq_length, path_img=self.path_img, vocabulary=self.vocabulary,
                                                         cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1, sep_token="[SEP]",
                                                         pad_on_left=False, pad_token=0, pad_token_segment_id=0, sequence_a_segment_id=0, mask_padding_with_zero=True,
                                                         crop_size=self.crop_size, ti_crop_size=self.ti_crop_size)
        else:
            feature = convert_one_msa_example_to_feature(example=example, label_list=self.label_list, tokenizer=self.tokenizer,
                                                         max_seq_length=self.max_seq_length, path_img=self.path_img, vocabulary=self.vocabulary,
                                                         cls_token_at_end=False, cls_token="<s>", cls_token_segment_id=0, sep_token="</s>",
                                                         pad_on_left=False, pad_token=1, pad_token_segment_id=0, sequence_a_segment_id=2, mask_padding_with_zero=True,
                                                         crop_size=self.crop_size, ti_crop_size=self.ti_crop_size)
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)
        img_feat = feature.img_feat
        label_ids = torch.tensor(feature.label_ids, dtype=torch.long)
        img_ti_feat = feature.img_ti_feat
        caption = torch.tensor(feature.caption)
        length = torch.tensor(feature.caption_len).long()
        input_len = torch.tensor(feature.input_len).long()

        return input_ids, input_mask, segment_ids, img_feat, label_ids, img_ti_feat, caption, length, input_len



class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='Training')
        >>> step = 2
        >>> pbar(step=step,info={'loss':20})
    '''

    def __init__(self, n_total, width=30, desc='Training', num_epochs=None):

        self.width = width
        self.n_total = n_total
        self.desc = desc
        self.start_time = time.time()
        self.num_epochs = num_epochs

    def reset(self):
        """Method to reset internal variables."""
        self.start_time = time.time()

    def _time_info(self, now, current):
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'
        return time_info

    def _bar(self, now, current):
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        return bar

    def epoch_start(self, current_epoch):
        sys.stdout.write("\n")
        if (current_epoch is not None) and (self.num_epochs is not None):
            sys.stdout.write(f"Epoch: {current_epoch}/{self.num_epochs}")
            sys.stdout.write("\n")

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        bar = self._bar(now, current)
        show_bar = f"\r{bar}" + self._time_info(now, current)
        if len(info) != 0:
            show_bar = f'{show_bar} ' + " [" + "-".join(
                [f' {key}={value:.4f} ' for key, value in info.items()]) + "]"
        if current >= self.n_total:
            show_bar += '\n'
        sys.stdout.write(show_bar)
        sys.stdout.flush()


class Vocab_Builder:
    def __init__(self, freq_threshold):

        # freq_threshold is to allow only words with a frequency higher
        # than the threshold

        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>",
                     3: "<UNK>"}  # index to string mapping
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2,
                     "<UNK>": 3}  # string to index mapping
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # Removing spaces, lower, general vocab related work
        return [token.text.lower() for token in nlp.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}  # dict to lookup for words
        idx = 4

        # FIXME better ways to do this are there
        for sentence in sentence_list:
            # print('sentence', sentence)
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if(frequencies[word] == self.freq_threshold):
                    # Include it
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    # Convert text to numericalized values
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)  # Get the tokenized text

        # Stoi contains words which passed the freq threshold. Otherwise, get the <UNK> token
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

    def denumericalize(self, token):
        text = [self.itos[token] if token in self.itos else self.itos[3]]
        return text

def get_vocabulary(examples):
    '''
        the vocab bulid for image caption
    '''
    logging.info("Constructing vocabulary for image caption")
    captions_list = []
    for example in examples:
        # captions = example.text_a.split(' ')
        captions_list.append(example.text_a)

    vocabulary = Vocab_Builder(freq_threshold=2)
    vocabulary.build_vocabulary(captions_list)
    return vocabulary

def get_msa_vocabulary(examples, path_img):
    '''
        the vocab bulid for image caption
    '''
    logging.info("Constructing vocabulary for image caption")
    captions_list = []
    for example in examples:
        text_name = path_img + '/' + str(example.img_id) + '.txt'
        # print("text_name:", text_name)
        with open(text_name, 'r', encoding='unicode_escape') as f:
            text = f.readlines()
        captions_list.append(text[0])
        # print("caption_list:", captions_list)

    vocabulary = Vocab_Builder(freq_threshold=2)
    vocabulary.build_vocabulary(captions_list)
    return vocabulary


def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2label, markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


class SeqEntityScore(object):
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + \
            precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(
                recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(
                label_path, self.id2label, self.markup)
            pre_entities = get_entities(pre_path, self.id2label, self.markup)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend(
                [pre_entity for pre_entity in pre_entities if pre_entity in label_entities])


def json_to_text(file_path, data):
    '''
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
