const { addon: ov } = require("openvino-node");

const { cv } = require("opencv-wasm");
const { getImageData, normalizeImageData } = require("./helper.js");

const modelXMLPath = "./model/yolov8s-cls.xml";
const imagePath = "./imgs/car.png";
const deviceName = "CPU";

main();

async function main() {
  const core = new ov.Core();
  const model = await core.readModel(modelXMLPath);
  const compiledModel = await core.compileModel(model, deviceName);

  const outputLayer = compiledModel.outputs[0];

  const imgData = await getImageData(imagePath);
  imgData.data = new Float32Array(imgData.data);
  normalizeImageData(imgData);

  // Use opencv-wasm to preprocess image.
  const originalImage = cv.matFromImageData(imgData);
  const image = new cv.Mat();
  // The MobileNet model expects images in RGB format.
  cv.cvtColor(originalImage, image, cv.COLOR_RGBA2RGB);
  // Resize to MobileNet image shape.
  cv.resize(image, image, new cv.Size(224, 224));

  const tensorData = new Float32Array(image.data);
  const tensor = new ov.Tensor(
    ov.element.f32,
    Int32Array.from([1, 3, 224, 224]),
    tensorData,
  );

  const inferRequest = compiledModel.createInferRequest();
  inferRequest.setInputTensor(tensor);
  inferRequest.infer();

  const resultInfer = inferRequest.getTensor(outputLayer);
  const resultIndex = resultInfer.data.indexOf(Math.max(...resultInfer.data));

  console.log("=== Result ===");
  console.log(`Index: ${resultIndex}`);

  const imagenetClassesMap = require("./assets/datasets/imagenet_class_index.json");
  const imagenetClasses = ["background", ...Object.values(imagenetClassesMap)];

  console.log(`Label: ${imagenetClasses[resultIndex][1]}`);
}
