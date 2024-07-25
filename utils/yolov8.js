const { nmsBoxes } = require('./dnn');
const { cv } = require('opencv-wasm');
const { Result, TYPES } = require('./result.js');

class YoloV8 {
  /**
   * 
   * @param {
   *  @param scales    scaling ratio h, scaling ratio h, height, width
   *  @param categNums score threshold
   *  @param scoreThreshold score threshold
   *  @param nmsThreshold   nms threshold
   * } options 
   */
  constructor(options) {
    this.scales = options.scales;
    this.categNums = options.categNums;
    this.scoreThreshold = 0.3;
    this.nmsThreshold = 0.5;
  }
  /**
   * Result process
   *
   * @param detect detection output
   * @param proto  segmentation output
   * @return
   */
  processDetResult(result) {
    // 检测数据
    let resultData = new cv.Mat(4 + this.categNums, 8400, cv.CV_32F);

    // let detectView = resultData.data;
    const dataPtr = resultData.data32F;
    for (let i = 0; i < result.length; i++) {
      dataPtr[i] = result[i];
    }
    cv.transpose(resultData, resultData);


    const positionBoxes = [];
    const classIds = [];
    const confidences = [];
    for (let i = 0; i < resultData.rows; i++) {
      const classesScores = resultData.row(i).colRange(4, 4 + this.categNums);
      let maxClassIdPoint = new cv.Point();
      let minClassIdPoint = new cv.Point();
      let maxScore = 0;
      let minScore = 0;
      const minMaxLocResult = cv.minMaxLoc(classesScores);
      maxClassIdPoint = minMaxLocResult.maxLoc;
      minClassIdPoint = minMaxLocResult.minLoc;
      maxScore = minMaxLocResult.maxVal;
      minScore = minMaxLocResult.minVal;

      if (maxScore > 0.25) {
        // const mask = resultData.row(i).colRange(4 + this.categNums, this.categNums + 36);
        const cx = resultData.data[i, 0];
        const cy = resultData.data[i, 1];
        const ow = resultData.data[i, 2];
        const oh = resultData.data[i, 3];
        const x = parseInt((cx - 0.5 * ow) * this.scales[0]);
        const y = parseInt((cy - 0.5 * oh) * this.scales[1]);
        const width = parseInt(ow * this.scales[0]);
        const height = parseInt(oh * this.scales[1]);
        const box = new cv.Rect();
        box.x = x;
        box.y = y;
        box.width = width;
        box.height = height;

        positionBoxes.push(box);
        classIds.push(parseInt(maxClassIdPoint.x));
        confidences.push(parseFloat(maxScore));
      }
    }
    const dnn = new cv.dnn_Net;
    const filterBoxes = new cv.dnn_Net(positionBoxes, confidences, this.scoreThreshold, this.nmsThreshold, classIds);
    const detResult = new Result(TYPES.TYPE_DETECTION);
    if (filterBoxes.length) {
      for (let i = 0; i < filterBoxes.Length; i++) {
        const index = filterBoxes[i];
        const box = positionBoxes[index];
        const rect = {
          x: box.x,
          y: box.y,
          width: box.width,
          height: box.height
        }
        detResult.add(confidences[index], rect, classIds[index])
      }
    }
    return detResult;
  }

  /**
   * Result process
   *
   * @param detect detection output
   * @param proto  segmentation output
   * @return
   */
  processSegResult(detect, proto) {
    // 检测数据
    let detectData = new cv.Mat(36 + this.categNums, 8400, cv.CV_32F);
    // detectData.setTo(0, 0, detect);
    let detectView = detectData.data;
    for (let i = 0; i < detectView.length; i++) {
      detectView[i] = detect[i];
    }
    cv.transpose(detectData, detectData);

    const protoData = new cv.Mat(32, 25600, cv.CV_32F);
    let proView = protoData.data;
    for (let i = 0; i < proView.length; i++) {
      proView[i] = proto[i];
    }

    const positionBoxes = [];
    const classIds = [];
    const confidences = [];
    const masks = [];
    for (let i = 0; i < detectData.rows; i++) {
      const classesScores = detectData.row(i).colRange(4, 4 + categNums);//GetArray(i, 5, classes_scores);
      let maxClassIdPoint = new cv.Point();
      let minClassIdPoint = new cv.Point();
      let maxScore = 0
      let minScore = 0
      cv.minMaxLoc(classesScores, minScore, maxScore, maxClassIdPoint, minClassIdPoint);
      // maxClassIdPoint = minMaxLocResult.maxLoc;
      // minClassIdPoint = minMaxLocResult.minLoc;
      // maxScore = minMaxLocResult.maxVal;
      // minScore = minMaxLocResult.minVal;

      if (maxScore > 0.25) {
        //Console.WriteLine(max_score);
        const mask = detectData.row(i).colRange(4 + categNums, categNums + 36);
        const cx = detectData.at(float.class, i, 0).getV();
        const cy = detectData.at(float.class, i, 1).getV();
        const ow = detectData.at(float.class, i, 2).getV();
        const oh = detectData.at(float.class, i, 3).getV();
        const x = parseInt((cx - 0.5 * ow) * this.scales[0]);
        const y = parseInt((cy - 0.5 * oh) * this.scales[1]);
        const width = parseInt(ow * this.scales[0]);
        const height = parseInt(oh * this.scales[1]);
        const box = new cv.Rect();
        box.x = x;
        box.y = y;
        box.width = width;
        box.height = height;

        positionBoxes.add(box);
        classIds.add(parseInt(maxClassIdPoint.x));
        confidences.add(parseFloat(maxScore));
        masks.add(mask);
      }
    }

    const filterBoxes = nmsBoxes(positionBoxes, confidences, 0.5, 0.5);
    if (filterBoxes.length) {
      for (let i = 0; i < filterBoxes.Length; i++) {
        const index = filterBoxes[i];
        const box = positionBoxes[index];
        console.log('box>>>', box);
        const boxX1 = Math.max(0, box.x);
        const boxY1 = Math.max(0, box.y);
        const boxX2 = Math.max(0, box.x + box.width);
        const boxY2 = Math.max(0, box.y + box.height);
        const originalMask = cv.multiply(masks[indx], protoData);
        for (let col = 0; col < originalMask.cols; col++) {
          originalMask.at(float.class, 0, col).setV(sigmoid(originalMask.at(float.class, 0, col).getV()));
        }
      }
    }
  }
};
module.exports = YoloV8;
