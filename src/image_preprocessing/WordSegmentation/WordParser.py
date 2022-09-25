from image_preprocessing.PicHandler import *
from utils.TextBlock import TextBlock
from image_preprocessing.WordSegmentation.FastSegmentation.FastSegmentator import FastSegmentator, FAST_MODE
from image_preprocessing.WordSegmentation.Segmentator import Segmentator


class WordParser:
    @staticmethod
    def parse(parse_method: Callable, img: PicHandler, resize_to: Tuple[int, int] = None, rescale_k: float = None,
              **args) -> List[TextBlock]:

        old_shape = img.get_image().shape
        if resize_to is None:
            resize_to = int(old_shape[0] * rescale_k), int(old_shape[1] * rescale_k)

        t_img = PicHandler(img.get_copy(), make_copy=False, is_colored=False)
        t_img.resize(resize_to)

        py, px = old_shape[0] / resize_to[0], old_shape[1] / resize_to[1]
        for k in args:
            args[k] /= np.sqrt(py * px)

        res: List[TextBlock] = parse_method(t_img.make_zero_one(), **args)

        for tb in res:
            tb.zone.multiply_coordinates(px, py)
            tb.zone.limit_bottom_right(old_shape[1] - 1, old_shape[0] - 1)
        return res


if __name__ == '__main__':
    import time

    ph = PicHandler('../test/test1.png')

    def pipeline(ph):
        ph.resize((850, 900))
        ph.apply_adaptive_bin_filter(w=0.15)
        ph.apply_filter(GAUSSIAN_FILTER, 3)
        ph.apply_adaptive_bin_filter(w=0.15)

    ph.exec_pipeline(pipeline)
    ph.show()

    start_time = time.time()

    tbs = WordParser.parse(Segmentator.parse_image_trivial, ph, MIN_LINE_DIST=0, MAX_LOCAL_SCOPE=20,
                           MIN_LINE_HEIGHT=1, MIN_WORDS_DIST=5, MIN_HOR_SUM=1, MIN_VERT_SUM=1, rescale_k=1)
    '''tbs = WordParser.parse(Segmentator.parse_image, ph, sensitivity=5, rescale_k=0.5, MIN_LINE_DIST=0,
                           MIN_LINE_HEIGHT=1, MIN_WORDS_DIST=5, MIN_HOR_SUM=2, MIN_VERT_SUM=2, MAX_LOCAL_SCOPE=20)'''

    #tbs = WordParser.parse(FastSegmentator.parse, ph, rescale_k=0.5, sens=5, MAX_LOCAL_SCOPE=20, PARSING_MODE=FAST_MODE)

    print(time.time() - start_time)
    p = PicHandler(ph.img, make_copy=True)
    for tb in tbs:
        p.draw_rect(tb.zone, 120)
    p.show()
