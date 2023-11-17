import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import ch.bildspur.vision.dependency.*;

import java.nio.file.Paths;
import java.nio.file.Path;

DeepVision vision;
MLSDNetwork network;

String url = "mlsd_512x512_large.onnx";
//String url = "mlsd_512x512_tiny.onnx";
//String url = "mlsd_320x320_large.onnx";
//String url = "mlsd_320x320_tiny.onnx";

void modelSetup() {
  vision = new DeepVision(this);
  
  url = sketchPath(new File("data", url).getPath());
  
  println("Loading model from " + url);
  Path model = Paths.get(url).toAbsolutePath();
  network = new MLSDNetwork(model);
  network.setup();
}

PImage modelInference(PImage img) { 
  println("Inferencing...");
  ImageResult result = network.run(img);
  PImage returnImg = result.getImage();
  println("...done!"); 
  
  return returnImg;
}
