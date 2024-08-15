const { addon: ov } = require('openvino-node');

const { cv } = require('opencv-wasm');
const { getImageData, hwcToNchw } = require('./helper.js');

main('./model/yolov8s-cls.xml', './imgs/car.png');

async function main(modelPath, imagePath) {
  //----------------- Step 1. Initialize OpenVINO Runtime Core -----------------
  console.log('Creating OpenVINO Runtime Core');
  const core = new ov.Core();

  //----------------- Step 2. Read a model -------------------------------------
  console.log(`Reading the model: ${modelPath}`);
  const model = await core.readModel(modelPath);

  //----------------- Step 3. Set up input -------------------------------------
  // Read input image
  const imgData = await getImageData(imagePath);

  // Use opencv-wasm to preprocess image.
  const originalImage = cv.matFromImageData(imgData);
  const image = new cv.Mat();
  // The MobileNet model expects images in RGB format.
  cv.cvtColor(originalImage, image, cv.COLOR_RGBA2RGB);
  // Resize to MobileNet image shape.
  cv.resize(image, image, new cv.Size(224, 224));

  const tensorData = new Float32Array(image.data);
  const data = hwcToNchw(tensorData, image.rows, image.cols, 3);

  const inputTensor = new ov.Tensor(ov.element.f32, Int32Array.from([1, 3, 224, 224]), data);


  //----------------- Step 5. Loading model to the device ----------------------
  console.log('Loading the model to the plugin');
  const compiledModel = await core.compileModel(model, 'CPU');

  //---------------- Step 6. Create infer request and do inference synchronously
  console.log('Starting inference in synchronous mode');
  const inferRequest = compiledModel.createInferRequest();
  inferRequest.setInputTensor(inputTensor);
  inferRequest.infer();

  //----------------- Step 7. Process output -----------------------------------
  const outputLayer = compiledModel.outputs[0];
  const resultInfer = inferRequest.getTensor(outputLayer);
  const resultIndex = resultInfer.data.indexOf(Math.max(...resultInfer.data));
  console.log("=== Result ===");
  console.log(`Index: ${resultIndex}`);
  const predictions = Array.from(resultInfer.data)
    .map((prediction, classId) => ({ prediction, classId }))
    .sort(({ prediction: predictionA }, { prediction: predictionB }) =>
      predictionA === predictionB ? 0 : predictionA > predictionB ? -1 : 1);
  // const imagenetClassesMap = require("./assets/datasets/imagenet_class_index.json");
  // const imagenetClasses = ["background", ...Object.values(imagenetClassesMap)];
  const imagenetData = require("./assets/datasets/yolov8-imagenet.json");
  const classNameMap = imagenetData.names;
  console.log(`Image path: ${imagePath}`);
  console.log('Top 10 results:');
  console.log('class_id probability');
  console.log('--------------------');
  predictions.slice(0, 10).forEach(({ classId, prediction }) =>
    console.log(`${classId}\t ${prediction.toFixed(7)}\t ${classNameMap[classId]}`),
  );
}
