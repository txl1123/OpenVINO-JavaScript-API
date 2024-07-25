const { addon: ov } = require('openvino-node');

const { cv } = require('opencv-wasm');
const { getImageData } = require('./helper.js');

main('./model/yolov8s-cls.xml', './imgs/car.jpeg');

async function main(modelPath, imagePath) {
  //----------------- Step 1. Initialize OpenVINO Runtime Core -----------------
  console.log('Creating OpenVINO Runtime Core');
  const core = new ov.Core();

  //----------------- Step 2. Read a model -------------------------------------
  console.log(`Reading the model: ${modelPath}`);
  const model = await core.readModel(modelPath);
  console.log('>>>model: ', model.inputs, model.outputs);

  //----------------- Step 3. Set up input -------------------------------------
  // Read input image
  const imgData = await getImageData(imagePath);
  try {
    let bboxes = new cv.MatOfRect2d();
    console.log(bboxes)
  } catch (e) {
    console.error(e)
  }
  // Use opencv-wasm to preprocess image.
  const originalImage = cv.matFromImageData(imgData);
  const image = new cv.Mat();
  // The MobileNet model expects images in RGB format.
  cv.cvtColor(originalImage, image, cv.COLOR_RGBA2RGB);

  const tensorData = new Float32Array(image.data);
  const shape = [1, image.rows, image.cols, 3];
  const inputTensor = new ov.Tensor(ov.element.f32, shape, tensorData);

  //----------------- Step 4. Apply preprocessing ------------------------------
  const _ppp = new ov.preprocess.PrePostProcessor(model);
  _ppp.input().tensor().setShape(shape).setLayout('NHWC');
  _ppp.input().preprocess().resize(ov.preprocess.resizeAlgorithm.RESIZE_LINEAR);
  _ppp.input().model().setLayout('NCHW');
  _ppp.output().tensor().setElementType(ov.element.f32);
  _ppp.build();

  //----------------- Step 5. Loading model to the device ----------------------
  console.log('Loading the model to the plugin');
  const compiledModel = await core.compileModel(model, 'AUTO');

  //---------------- Step 6. Create infer request and do inference synchronously
  console.log('Starting inference in synchronous mode');
  const inferRequest = compiledModel.createInferRequest();
  inferRequest.setInputTensor(inputTensor);
  inferRequest.infer();

  //----------------- Step 7. Process output -----------------------------------
  const outputLayer = compiledModel.outputs[0];
  const resultInfer = inferRequest.getTensor(outputLayer);
  const predictions = Array.from(resultInfer.data)
    .map((prediction, classId) => ({ prediction, classId }))
    .sort(({ prediction: predictionA }, { prediction: predictionB }) =>
      predictionA === predictionB ? 0 : predictionA > predictionB ? -1 : 1);

  console.log(`Image path: ${imagePath}`);
  console.log('Top 10 results:');
  console.log('class_id probability');
  console.log('--------------------');
  predictions.slice(0, 10).forEach(({ classId, prediction }) =>
    console.log(`${classId}\t ${prediction.toFixed(7)}`),
  );
}
