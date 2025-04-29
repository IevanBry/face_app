import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class EmotionDetectorScreen extends StatefulWidget {
  const EmotionDetectorScreen({super.key});

  @override
  State<EmotionDetectorScreen> createState() => _EmotionDetectorScreenState();
}

class _EmotionDetectorScreenState extends State<EmotionDetectorScreen> {
  File? _selectedImage;

  Future<void> _pickImageFromGallery() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
      });
    }
  }

  Future<void> _pickImageFromCamera() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.camera);

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 40.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(
                Icons.blur_circular_outlined,
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
              const SizedBox(height: 60),
              OptionButton(
                icon: Icons.photo,
                label: 'Choose From\nGallery',
                gradient: const LinearGradient(
                  colors: [Color(0xFF9C4521), Color(0xFFD98248)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                onTap: _pickImageFromGallery,
              ),
              const SizedBox(height: 30),
              OptionButton(
                icon: Icons.photo_camera,
                label: 'Take a new\nphoto / video',
                gradient: const LinearGradient(
                  colors: [Color(0xFFD9A144), Color(0xFFF5C980)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                onTap: _pickImageFromCamera,
              ),
              const SizedBox(height: 30),
              if (_selectedImage != null)
                Expanded(
                  child: Center(
                    child: Image.file(_selectedImage!),
                  ),
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
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                ),
              ),
            ),
            const Icon(Icons.arrow_forward, color: Colors.white),
          ],
        ),
      ),
    );
  }
}