import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class EmotionDetectorScreen extends StatefulWidget {
  const EmotionDetectorScreen({super.key});

  @override
  State<EmotionDetectorScreen> createState() => _EmotionDetectorScreenState();
}

class _EmotionDetectorScreenState extends State<EmotionDetectorScreen> {
  File? _selectedImage;
  late Interpreter _interpreter;
  List<String> _labels = [];
  String _predictionResult = "";
  Rect? _faceBoundingBox;
  img.Image? _originalImage;

  @override
  void initState() {
    super.initState();
    _loadModelAndLabels();
  }

  Future<void> _loadModelAndLabels() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/best_float32.tflite');
      final labelData = await rootBundle.loadString('assets/label.txt');
      _labels = labelData.split('\n');
      print("Model and labels loaded.");
    } catch (e) {
      print("Error loading model/labels: $e");
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      final imageFile = File(pickedFile.path);
      setState(() {
        _selectedImage = imageFile;
        _faceBoundingBox = null;
      });
      await _detectAndClassify(imageFile);
    }
  }

  Future<void> _detectAndClassify(File imageFile) async {
    final inputImage = InputImage.fromFile(imageFile);
    final faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        performanceMode: FaceDetectorMode.accurate,
        enableLandmarks: false,
        enableContours: false,
      ),
    );

    final faces = await faceDetector.processImage(inputImage);
    await faceDetector.close();

    if (faces.isEmpty) {
      setState(() => _predictionResult = "Tidak ada wajah terdeteksi.");
      return;
    }

    final imageBytes = await imageFile.readAsBytes();
    _originalImage = img.decodeImage(imageBytes);

    final face = faces[0].boundingBox;

    final croppedFace = img.copyCrop(
      _originalImage!,
      x: face.left.toInt().clamp(0, _originalImage!.width - 1),
      y: face.top.toInt().clamp(0, _originalImage!.height - 1),
      width: face.width.toInt().clamp(0, _originalImage!.width),
      height: face.height.toInt().clamp(0, _originalImage!.height),
    );

    final resizedFace = img.copyResize(croppedFace, width: 224, height: 224);
    final input = _imageToInput(resizedFace);

    final output = List.filled(
      _labels.length,
      0.0,
    ).reshape([1, _labels.length]);
    _interpreter.run([input], output);

    final prediction = output[0] as List<double>;
    final maxIndex = prediction.indexWhere(
      (e) => e == prediction.reduce((a, b) => a > b ? a : b),
    );

    setState(() {
      _faceBoundingBox = face;
      _predictionResult =
          "${_labels[maxIndex]} (${(prediction[maxIndex] * 100).toStringAsFixed(2)}%)";
    });
  }

  List<List<List<double>>> _imageToInput(img.Image image) {
    return List.generate(224, (y) {
      return List.generate(224, (x) {
        final pixel = image.getPixel(x, y);
        return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(
                Icons.emoji_emotions,
                color: Colors.orangeAccent,
                size: 40,
              ),
              const SizedBox(height: 20),
              const Text(
                'Emotion Detector',
                style: TextStyle(color: Colors.white, fontSize: 24),
              ),
              const SizedBox(height: 30),
              Row(
                children: [
                  Expanded(
                    child: OptionButton(
                      icon: Icons.photo,
                      label: 'Gallery',
                      onTap: () => _pickImage(ImageSource.gallery),
                    ),
                  ),
                  const SizedBox(width: 20),
                  Expanded(
                    child: OptionButton(
                      icon: Icons.camera_alt,
                      label: 'Camera',
                      onTap: () => _pickImage(ImageSource.camera),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 30),
              if (_selectedImage != null)
                Expanded(
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      return Stack(
                        children: [
                          Image.file(_selectedImage!, fit: BoxFit.contain, width: constraints.maxWidth),
                          if (_faceBoundingBox != null && _originalImage != null)
                            Positioned.fill(
                              child: CustomPaint(
                                painter: FaceBoxPainter(
                                  _faceBoundingBox!,
                                  originalSize: Size(
                                    _originalImage!.width.toDouble(),
                                    _originalImage!.height.toDouble(),
                                  ),
                                ),
                              ),
                            ),
                        ],
                      );
                    },
                  ),
                ),
              const SizedBox(height: 20),
              Text(
                _predictionResult,
                style: const TextStyle(color: Colors.white, fontSize: 18),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class OptionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const OptionButton({
    super.key,
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Ink(
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF9C4521), Color(0xFFD98248)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(16),
        ),
        padding: const EdgeInsets.all(24),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: Colors.white),
            const SizedBox(width: 12),
            Text(
              label,
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}

class FaceBoxPainter extends CustomPainter {
  final Rect faceRect;
  final Size originalSize;

  FaceBoxPainter(this.faceRect, {required this.originalSize});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    // Scale bounding box to displayed image size
    double scaleX = size.width / originalSize.width;
    double scaleY = size.height / originalSize.height;

    Rect scaledRect = Rect.fromLTRB(
      faceRect.left * scaleX,
      faceRect.top * scaleY,
      faceRect.right * scaleX,
      faceRect.bottom * scaleY,
    );

    canvas.drawRect(scaledRect, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}