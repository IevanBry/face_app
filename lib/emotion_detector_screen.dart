import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;

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

  @override
  void initState() {
    super.initState();
    _loadModelAndLabels();
  }

  Future<void> _loadModelAndLabels() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/best_float32.tflite');
      print("Model loaded!");

      // Load label dari file
      final labelData = await rootBundle.loadString('assets/label.txt');
      _labels = labelData.split('\n');
    } catch (e) {
      print("Failed to load model or labels: $e");
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      final file = File(pickedFile.path);
      setState(() {
        _selectedImage = file;
      });
      await _classifyEmotion(file);
    }
  }

  Future<void> _classifyEmotion(File image) async {
    final inputImage = await _preprocessImage(image);

    final input = [inputImage];

    final output = List.filled(
      _labels.length,
      0.0,
    ).reshape([1, _labels.length]);

    _interpreter.run(input, output);

    final predictions = output[0] as List<double>;
    final maxIndex = predictions.indexWhere(
      (e) => e == predictions.reduce((a, b) => a > b ? a : b),
    );
    final label = _labels[maxIndex];
    final confidence = predictions[maxIndex];

    setState(() {
      _predictionResult = "$label (${(confidence * 100).toStringAsFixed(2)}%)";
    });

    print("Prediction: $_predictionResult");
  }

  Future<List<List<List<double>>>> _preprocessImage(File image) async {
    final imageBytes = await image.readAsBytes();
    img.Image? imageRaw = img.decodeImage(imageBytes);
    imageRaw = img.copyResize(imageRaw!, width: 224, height: 224);

    return List.generate(224, (y) {
      return List.generate(224, (x) {
        final pixel = imageRaw!.getPixel(x, y);
        final r = pixel.r / 255.0;
        final g = pixel.g / 255.0;
        final b = pixel.b / 255.0;
        return [r, g, b];
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 40),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(
                Icons.emoji_emotions,
                color: Colors.orangeAccent,
                size: 40,
              ),
              const SizedBox(height: 24),
              const Text(
                'Emotion Detector',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 40),
              OptionButton(
                icon: Icons.photo,
                label: 'Choose From\nGallery',
                gradient: const LinearGradient(
                  colors: [Color(0xFF9C4521), Color(0xFFD98248)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                onTap: () => _pickImage(ImageSource.gallery),
              ),
              const SizedBox(height: 30),
              OptionButton(
                icon: Icons.camera_alt,
                label: 'Take a new\nphoto',
                gradient: const LinearGradient(
                  colors: [Color(0xFFD9A144), Color(0xFFF5C980)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                onTap: () => _pickImage(ImageSource.camera),
              ),
              const SizedBox(height: 30),
              if (_selectedImage != null) ...[
                Expanded(child: Center(child: Image.file(_selectedImage!))),
                Text(
                  _predictionResult,
                  style: const TextStyle(color: Colors.white, fontSize: 18),
                  textAlign: TextAlign.center,
                ),
              ],
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
  final Gradient gradient;
  final VoidCallback onTap;

  const OptionButton({
    super.key,
    required this.icon,
    required this.label,
    required this.gradient,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Ink(
        decoration: BoxDecoration(
          gradient: gradient,
          borderRadius: BorderRadius.circular(16),
        ),
        padding: const EdgeInsets.all(24),
        child: Row(
          children: [
            Icon(icon, color: Colors.white),
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                label,
                style: const TextStyle(color: Colors.white, fontSize: 16),
              ),
            ),
            const Icon(Icons.arrow_forward, color: Colors.white),
          ],
        ),
      ),
    );
  }
}
