import uuid


class Face:
    def __init__(self, x, y, width, height, crop):
        self.id            = str(uuid.uuid4())
        self.x             = x
        self.y             = y
        self.width         = width
        self.height        = height
        self.crop          = crop
        self.is_new_person = False
        self.person_id     = None
        self.location      = None
        self.timestamp     = None
        self.features      = None

    def iou(self, other):
        b1x1 = self.x
        b1y1 = self.y
        b1x2 = self.x + self.width
        b1y2 = self.y + self.height

        b2x1 = other.x
        b2y1 = other.y
        b2x2 = other.x + other.width
        b2y2 = other.y + other.height

        x_left   = max(b1x1, b2x1)
        y_top    = max(b1y1, b2y1)
        x_right  = min(b1x2, b2x2)
        y_bottom = min(b1y2, b2y2)

        if x_right < x_left or y_bottom < y_top:
            return 0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bb1_area = (b1x2 - b1x1) * (b1y2 - b1y1)
        bb2_area = (b2x2 - b2x1) * (b2y2 - b2y1)

        return intersection_area / (bb1_area + bb2_area - intersection_area)
