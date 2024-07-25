/**
 * Yolov8 Key point data
 */
const TYPES = {
  TYPE_DETECTION: 0,
  TYPE_SEGMENTATION: 1,
  KEY_KEY_POINT_PREDICTION: 2,
}

class Result {
  constructor(options) {
    // 结果类型，默认是det
    this.type = options.type || TYPES.TYPE_DETECTION;
    this.classes = [];
    this.scores = [];
    this.rects = [];
    this.masks = [];
    this.poses = [];
  }

  /**
   * object detection
   *
   * @param score Predictiveness scores
   * @param rect  Identification box
   * @param cla   Identification class
   */
  add({ score, rect, cla, mask, pose }) {
    if (score) {
      this.scores.push(score);
    }
    if (rect) {
      this.rects.push(rect);
    }
    if (cla) {
      this.classes.push(cla);
    }
    if (mask) {
      this.masks.push(mask);
    }
    if (pose) {
      this.poses.add(pose);
    }
  }

  getType() {
    return this.type;
  }

  /**
   * Get Result Length
   *
   * @return
   */
  getLength() {
    return this.scores.length();
  }
}
module.exports = {
  Result,
  TYPES
};