from __future__ import annotations
from PicHandler import *
from BitMatrix import BitMatrix, get_crop_index
from image_preprocessing.WriteHelper import WriteHelper
from SegAnalyzer import SegAnalyzer
from utils.geometry import Point, Rect
import pickle
import os

IMG_PATH = 'D:\\projects\\datasets\\letters\\zip_letters\\letters'
LABEL_FILE = 'D:\\projects\\datasets\\letters\\letters3.txt'
DEFAULT_SAVE_NAME = 'character_examples'


class RelationDatabase:
    relation_set: Dict[str, float]
    CONNECTION_PART = 0.1

    def __init__(self, initialize: bool = True):
        if initialize:
            self.__create_relation_set()

    @staticmethod
    def __label_file_iterator():
        file = open(LABEL_FILE)
        file.readline()
        for line in file.readlines():
            fields = line.split(',')
            yield fields[1], fields[2]
        file.close()

    def __create_relation_set(self) -> None:
        print("Calculating relations for character examples...")
        def check_criterion(matrix: np.ndarray) -> bool:
            shape = matrix.shape
            tsum = (matrix == 0).sum()
            # print(tsum, shape[0] * shape[1])
            return tsum > 0.1 * shape[0] * shape[1]  # вроде очень точный порог адекватности

        self.relation_set = dict()
        variance = dict()
        mask_set = dict()
        prev_label = None

        for filename in os.listdir(IMG_PATH):
            parsed = filename.split('_')
            is_upper = int(parsed[0]) == 1
            if is_upper:
                label = WriteHelper.num_to_char(int(parsed[1]))
            else:
                label = WriteHelper.num_to_char(int(parsed[1]) + 33)

            if label not in mask_set.keys():
                if not (prev_label is None):
                    # вычислим среднее отношение ширины символа к высоте
                    dimensions = np.array([[_mask.matrix.shape[1], _mask.matrix.shape[0]]
                                           for _mask in mask_set[prev_label]])
                    self.relation_set[prev_label] = np.mean(dimensions[:, 0]) / np.mean(dimensions[:, 1])
                    variance[prev_label] = np.var(dimensions[:, 0] / dimensions[:, 1])

                    print(prev_label, self.relation_set[prev_label], variance[prev_label])

                mask_set[label] = []

            ph = PicHandler(IMG_PATH + '\\' + filename)

            ph.apply_adaptive_bin_filter()
            mask = BitMatrix(ph)

            if check_criterion(mask.matrix):
                mask_set[label].append(mask)
                prev_label = label

        dimensions = np.array([[_mask.matrix.shape[1], _mask.matrix.shape[0]]
                               for _mask in mask_set[prev_label]])
        self.relation_set[prev_label] = np.mean(dimensions[:, 0]) / np.mean(dimensions[:, 1])
        variance[prev_label] = np.var(dimensions[:, 0] / dimensions[:, 1])
        print(prev_label, self.relation_set[prev_label], variance[prev_label])

        print('variance:', variance)
        # print(len(self.mask_set.values()))
        # view_image(mask.mask)

    def __align_borders(self, word: str, img: np.ndarray) -> List[Tuple[int, int]]:
        # возвращает расположение горизонтальных границ между символами

        def calc_coef(_key) -> float:
            if WriteHelper.have_script(_key, True) or WriteHelper.have_script(_key, False):
                return 1.9
            return 1

        res = []
        #symb_keys = [WriteHelper.char_to_num(symb) for symb in word]
        ws = np.array([self.relation_set[key] * calc_coef(key) for key in word])
        ws /= ws.mean() * len(ws)

        dx = 0
        for w in ws:
            word_w = int(img.shape[1] * w)
            indent = word_w * RelationDatabase.CONNECTION_PART / 2
            if w != ws[-1]:
                res.append((int(dx + indent), int(dx + word_w - indent)))
            else:
                res.append((int(dx + indent), dx + word_w))
            dx += word_w

        return res

    def align_rects(self, word: str, img: np.ndarray) -> List[Tuple[Tuple[int, int], str]]:
        # для слова и изображения с ним размечает границы символов
        borders_x = self.__align_borders(word, img)

        return [(borders_x[i], word[i]) for i in range(len(word))]

    def save_mask_set(self, fname: str = DEFAULT_SAVE_NAME) -> None:
        file = open(fname, 'wb')
        pickle.dump(self.relation_set, file)
        file.close()

    @staticmethod
    def load_mask_set(fname: str = DEFAULT_SAVE_NAME) -> RelationDatabase:
        mf = RelationDatabase(False)
        file = open(fname, 'rb')
        mf.relation_set = pickle.load(file)
        file.close()
        return mf


if __name__ == '__main__':
    #mf = MaskDatabase()
    #print(len(mf.relation_set.values()))
    #mf.save_mask_set()
    mf = RelationDatabase.load_mask_set()
    #print(mf.relation_set)
    fnames = [('0_1_19.png', 'долго'), ('0_0_62.jpg', 'человек')]
    for fname, word in fnames:
        ph = PicHandler(fname)
        #ph = PicHandler('0_0_3.png')
        ph.apply_adaptive_bin_filter()
        bm = BitMatrix(ph)
        #view_image(mf.mask_set['1'][0].matrix)
        view_image(bm.matrix)

        res = mf.align_rects(word, bm.matrix)
        new_img = PicHandler(bm.matrix.copy())
        for zone, symb in res:
            rect = Rect(Point(zone[0], 0), Point(zone[1], new_img.img.shape[0] - 1))
            print(symb, rect)
            new_img.draw_rect(rect, 120)
        new_img.show()

    #res = mf.relation_set['1'][0].put_on(BitMatrix(ph), 'и докучных', 'и', 0)
    #res = 255 - res.astype(np.uint8)
    #view_image(res)
