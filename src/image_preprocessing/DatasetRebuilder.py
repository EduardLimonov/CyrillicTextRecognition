from __future__ import annotations
from RelationDatabase import *
from collections import deque
import json
import os
import pickle
from BitMatrix import BitMatrix


class DatasetRebuilder:
    database: RelationDatabase = RelationDatabase.load_mask_set()
    MIN_WORD_SPACE = 10

    @staticmethod
    def handle_raw_image(img: np.ndarray, text: str) -> List[Tuple[np.ndarray, List[Tuple[Tuple[int, int], str]]]]:
        # [(imageN, <list of symbol positions -- Tuples[Tuple[int, int], str]>)]

        def split_to_words(_img: np.ndarray, n_words: int, MIN_WORD_SIZE = 25, AREA = 25) -> List[np.ndarray]:
            # вычислим минимумы локальных сумм, они соответствуют пробелам
            # AREA <= MIN_WORD_SIZE

            if n_words == 1:
                return [_img]

            vert_sums = _img.sum(axis=0)
            r = np.zeros_like(vert_sums)
            for vert in range(MIN_WORD_SIZE, len(vert_sums)):
                left = vert - AREA
                right = min(len(vert_sums), vert + AREA)

                add = 0 if vert + AREA <= len(vert_sums) else 255 * (vert + AREA - len(vert_sums)) * _img.shape[0]
                r[vert] = vert_sums[left: right].sum() + add

            r_sorted = sorted(r, reverse=True)
            maximums = []
            deleted = 0
            for i in range(MIN_WORD_SIZE, len(r)):
                if r[i] in r_sorted[: n_words + deleted - 1]:
                    do_append = True
                    for idx in maximums:
                        if idx < i < idx + MIN_WORD_SIZE:
                            do_append = False
                            break

                    if do_append:
                        maximums.append(i)
                    else:
                        deleted += 1
                if len(maximums) == n_words - 1:
                    break

            res = []
            left = 0
            for next_min in maximums:
                res.append(BitMatrix(_img[:, left: next_min]).matrix)
                left = next_min
            res.append(BitMatrix(_img[:, left: ]).matrix)
            return res

        res = []
        words = text.split()
        word_imgs = split_to_words(img, len(words))
        for i in range(len(words)):
            word, img = words[i], word_imgs[i]
            if WriteHelper.has_punctuation(word):
                # TODO normal punctuation handler
                # img = split_to_words(img, 2, PUNC_SIZE, PUNC_AREA)[0]
                word = ''.join([s for s in word if not WriteHelper.has_punctuation(s)])

            rects = DatasetRebuilder.database.align_rects(word, img)
            res.append((img, rects))

        return res

    @staticmethod
    def read_json(ann_path_name: str) -> Tuple[str, str]:
        # пара <расшифровка, имя файла>
        with open(ann_path_name, 'rb') as file:
            data = json.loads(file.readline())
            return data["description"], data["name"]

    @staticmethod
    def save_data(data, path, name):
        file = open(path + '\\' + name, 'wb')
        pickle.dump(data, file)
        file.close()

    @staticmethod
    def rebuild(path: str, path_to_save: str) -> None:
        img_path = path + '\\' + 'img\\'
        ann_path = path + '\\' + 'ann'

        fname_idx = 0
        res: List[Tuple[np.ndarray, List[Tuple[Tuple[int, int], str]]]] = []  # пары вход -- выход
        err_files = []
        for filename in os.listdir(ann_path):
            word, img_name = DatasetRebuilder.read_json(ann_path + '\\' + filename)

            ph = PicHandler(img_path + img_name + '.jpg')
            ph.apply_adaptive_bin_filter()

            try:
                new_data = DatasetRebuilder.handle_raw_image(BitMatrix(ph.img).matrix, word)
            except:
                err_files.append(img_name)
                continue

            print(fname_idx, 'handled: %s' % img_name)
            # DatasetRebuilder.save_data(d, path_to_save, str(fname_idx))
            res += new_data
            if fname_idx % 10000 == 0:
                DatasetRebuilder.save_data(res, path_to_save, 'my_dataset' + str(fname_idx))
                res = []
            fname_idx += len(new_data)

        DatasetRebuilder.save_data(res, path_to_save, 'my_dataset' + str(fname_idx))
        print(len(err_files))
        print(err_files)


if __name__ == '__main__':

    def punc_test():
        ph = PicHandler('D:\\projects\\datasets\\HKR\\20200923_Dataset_Words_Public\\img\\0_0_6.jpg')
        ph.apply_adaptive_bin_filter()
        ph.show()
        res = DatasetRebuilder.handle_raw_image(
            ph.img,
            'Шёл человек.'
        )

        for img, rects in res:
            ph = PicHandler(img)
            for r, s in rects:
                ph.draw_rect(Rect(Point(r[0], 0), Point(r[1], 2)))
            ph.show()


    def rand_data_test():
        import random
        file = open('D:\\projects\\datasets\\align\\my_dataset10000', 'rb')
        a: List[Tuple[np.ndarray, List[Tuple[Rect, str]]]] = pickle.load(file)
        img, rects = random.choice(a)
        ph = PicHandler(img)
        for rect in rects:
            r = Rect(Point(rect[0][0], 0), Point(rect[0][1], img.shape[0] - 1))
            ph.draw_rect(r)
            print(rect[0], rect[1])
        ph.show()


    #punc_test()
    rand_data_test()
    exit(0)
    import time

    start_time = time.time()
    DatasetRebuilder.rebuild('D:\\projects\\datasets\\HKR\\20200923_Dataset_Words_Public',
                             'D:\\projects\\datasets\\align')
    print(time.time() - start_time, 'seconds')

    #file = open('D:\\projects\\datasets\\align\\my_dataset0', 'rb')
    #a = pickle.load(file)
    #print(a)
