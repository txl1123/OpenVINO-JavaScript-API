function nmsBoxes(boxes, scores, scoreThreshold, nmsThreshold, topK) {
  // 对分数进行降序排序，并记录原始索引
  const indices = [...Array(scores.length).keys()].sort((a, b) => scores[b] - scores[a]);

  // 只保留前topK个最高分的边界框
  if (topK > 0 && topK < indices.length) {
    indices.splice(topK);
  }

  // 初始化一个空数组来存储筛选后的边界框
  const filteredBoxes = [];

  while (indices.length > 0) {
    // 选择具有最高分数的边界框的索引
    const maxIndex = indices.shift();

    // 如果分数低于scoreThreshold，则跳出循环
    if (scores[maxIndex] < scoreThreshold) {
      break;
    }

    // 将当前边界框添加到筛选后的数组中
    filteredBoxes.push(boxes[maxIndex]);

    // 移除与当前边界框重叠的其他边界框
    indices = indices.filter((index) => {
      const iou = computeIoU(boxes[maxIndex], boxes[index]);
      return iou <= nmsThreshold;
    });
  }

  return filteredBoxes;
}

function computeIoU(box1, box2) {
  const x1 = Math.max(box1.x, box2.x);
  const y1 = Math.max(box1.y, box2.y);
  const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
  const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

  const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const unionArea = box1.width * box1.height + box2.width * box2.height - intersectionArea;

  return intersectionArea / unionArea;
}
module.exports = {
  nmsBoxes
}