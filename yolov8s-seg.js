const { addon: ov } = require('openvino-node');

const { cv } = require('opencv-wasm');
const { getImageData } = require('./helper.js');
const YoloV8 = require('./utils/yolov8.js');

main('./model/yolov8s-seg.xml', './imgs/car.jpeg');

async function main(modelPath, imagePath) {
  //----------------- Step 1. Initialize OpenVINO Runtime Core -----------------
  console.log('Creating OpenVINO Runtime Core');
  const core = new ov.Core();

  //----------------- Step 2. Read a model -------------------------------------
  console.log(`Reading the model: ${modelPath}`);
  const model = await core.readModel(modelPath);
  console.log('>>>model: ', model.inputs, model.outputs);
  if (model.inputs.length !== 1)
    throw new Error('Sample supports only single input topologies');

  // if (model.outputs.length !== 1)
  //   throw new Error('Sample supports only single output topologies');

  //----------------- Step 3. Set up input -------------------------------------
  // Read input image
  const imgData = await getImageData(imagePath);

  // Use opencv-wasm to preprocess image.
  const originalImage = cv.matFromImageData(imgData);
  const image = new cv.Mat();
  // The MobileNet model expects images in RGB format.
  cv.cvtColor(originalImage, image, cv.COLOR_RGBA2RGB);

  const maxImageLength = image.cols > image.rows ? image.cols : image.rows;
  const maxImage = cv.Mat.zeros(maxImageLength, maxImageLength, cv.CV_8U);
  // const roi = cv.Rect(0, 0, image.cols, image.rows);
  const copymat = new cv.Mat(image.rows, image.cols, cv.CV_8U);
  image.copyTo(copymat);
  // cv.resize(max_image, max_image, new Size(640, 640))
  const size = maxImageLength / 640;
  const factors = [size, size, image.rows, image.cols];

  // const tensorData = new Float32Array(image.data);
  // const shape = [1, image.rows, image.cols, 3];
  // const inputTensor = new ov.Tensor(ov.element.f32, shape, tensorData);

  //----------------- Step 4. Apply preprocessing ------------------------------
  // const _ppp = new ov.preprocess.PrePostProcessor(model);
  // _ppp.input().tensor().setShape(shape).setLayout('NHWC');
  // _ppp.input().preprocess().resize(ov.preprocess.resizeAlgorithm.RESIZE_LINEAR);
  // _ppp.input().model().setLayout('NCHW');
  // _ppp.output().tensor().setElementType(ov.element.f32);
  // _ppp.build();

  //----------------- Step 5. Loading model to the device ----------------------
  console.log('Loading the model to the plugin');
  const compiledModel = await core.compileModel(model, 'AUTO');

  //---------------- Step 6. Create infer request and do inference synchronously
  console.log('Starting inference in synchronous mode');
  const inferRequest = compiledModel.createInferRequest();
  // const tensor = inferRequest.getTensor();
  const itensor = inferRequest.getInputTensor()
  const inputShape = itensor.getShape();
  const inputMat = cv.blobFromImage(maxImage, 1 / 255, new cv.Size(inputShape[2], inputShape[3]), new cv.Scalar(0), true, false);
  const inputData = new Float32Array(image.data);
  // inputMat.copyTo(inputData);
  // inputData.a
  const shape = [1, inputShape[2], inputShape[3], 3];
  const inputTensor = new ov.Tensor(ov.element.f32, shape, inputData);
  // inputTensor.setInputTensor
  inferRequest.setInputTensor(inputTensor);
  inferRequest.infer();

  //----------------- Step 7. Process output -----------------------------------
  const outputDet = compiledModel.outputs[0];
  const detResultInfer = inferRequest.getTensor(outputDet);
  const predictionsDet = Array.from(detResultInfer.data);

  const outputPro = compiledModel.outputs[0];
  const proResultInfer = inferRequest.getTensor(outputPro);
  const predictionsPro = Array.from(proResultInfer.data);

  const yoloProcess = new YoloV8(factors, 80);
  const res = yoloProcess.processSegResult(predictionsDet, predictionsPro);
  console.log(res);

  console.log(`Image path: ${imagePath}`);
  console.log('Top 10 results:');
  console.log('class_id probability');
  console.log('--------------------');
  predictions.slice(0, 10).forEach(({ classId, prediction }) =>
    console.log(`${classId}\t ${prediction.toFixed(7)}`),
  );

  console.log('\nThis sample is an API example, for any performance '
    + 'measurements please use the dedicated benchmark_app tool');
}