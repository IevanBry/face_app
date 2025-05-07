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
  List<Rect> _faceBoundingBoxes = [];
  List<String> _predictionResults = [];
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
        _faceBoundingBoxes = [];
        _predictionResults = [];
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
      setState(() {
        _predictionResults = ["Tidak ada wajah terdeteksi."];
        _faceBoundingBoxes = [];
      });
      return;
    }

    final imageBytes = await imageFile.readAsBytes();
    _originalImage = img.decodeImage(imageBytes);

    List<Rect> boundingBoxes = [];
    List<String> predictions = [];

    for (var face in faces) {
      final box = face.boundingBox;

      final croppedFace = img.copyCrop(
        _originalImage!,
        x: box.left.toInt().clamp(0, _originalImage!.width - 1),
        y: box.top.toInt().clamp(0, _originalImage!.height - 1),
        width: box.width.toInt().clamp(1, _originalImage!.width - box.left.toInt()),
        height: box.height.toInt().clamp(1, _originalImage!.height - box.top.toInt()),
      );

      final resizedFace = img.copyResize(croppedFace, width: 224, height: 224);
      final input = _imageToInput(resizedFace);

      final output = List.filled(_labels.length, 0.0).reshape([1, _labels.length]);
      _interpreter.run([input], output);

      final prediction = output[0] as List<double>;
      final maxIndex = prediction.indexWhere(
        (e) => e == prediction.reduce((a, b) => a > b ? a : b),
      );

      boundingBoxes.add(box);
      predictions.add("${_labels[maxIndex]} (${(prediction[maxIndex] * 100).toStringAsFixed(2)}%)");
    }

    setState(() {
      _faceBoundingBoxes = boundingBoxes;
      _predictionResults = predictions;
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
                          if (_faceBoundingBoxes.isNotEmpty && _originalImage != null)
                            Positioned.fill(
                              child: CustomPaint(
                                painter: FaceBoxPainter(
                                  faceRects: _faceBoundingBoxes,
                                  labels: _predictionResults,
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
              ..._predictionResults.map((res) => Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  res,
                  style: const TextStyle(color: Colors.white, fontSize: 16),
                  textAlign: TextAlign.center,
                ),
              )),
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
  final List<Rect> faceRects;
  final List<String> labels;
  final Size originalSize;

  FaceBoxPainter({
    required this.faceRects,
    required this.labels,
    required this.originalSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    double scaleX = size.width / originalSize.width;
    double scaleY = size.height / originalSize.height;

    for (int i = 0; i < faceRects.length; i++) {
      final rect = faceRects[i];
      final scaledRect = Rect.fromLTRB(
        rect.left * scaleX,
        rect.top * scaleY,
        rect.right * scaleX,
        rect.bottom * scaleY,
      );

      canvas.drawRect(scaledRect, paint);

      final textSpan = TextSpan(
        text: labels[i],
        style: const TextStyle(color: Colors.black, fontSize: 16),
      );

      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      final offset = Offset(scaledRect.left, scaledRect.bottom + 4);
      final bgRect = Rect.fromLTWH(
        offset.dx - 4,
        offset.dy - 2,
        textPainter.width + 8,
        textPainter.height + 4,
      );

      final bgPaint = Paint()..color = Colors.white.withOpacity(0.8);
      canvas.drawRRect(
        RRect.fromRectAndRadius(bgRect, const Radius.circular(4)),
        bgPaint,
      );

      textPainter.paint(canvas, offset);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}