import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;
import 'package:video_thumbnail/video_thumbnail.dart';
import 'package:video_player/video_player.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image_gallery_saver_plus/image_gallery_saver_plus.dart';

class EmotionDetectorScreen extends StatefulWidget {
  const EmotionDetectorScreen({Key? key}) : super(key: key);

  @override
  State<EmotionDetectorScreen> createState() => _EmotionDetectorScreenState();
}

class _EmotionDetectorScreenState extends State<EmotionDetectorScreen> {
  late Interpreter _interpreter;
  List<String> _labels = [];

  late FaceDetector _faceDetector;

  File? _selectedImage;
  img.Image? _originalImage;
  List<Rect> _faceBoxes = [];
  List<String> _predictions = [];

  File? _selectedVideo;
  bool _isProcessing = false;

  List<img.Image> _processedFrames = [];
  List<String> _frameLabels = [];

  @override
  void initState() {
    super.initState();
    _loadModelAndLabels();
    _initFaceDetector();
  }

  Future<void> _loadModelAndLabels() async {
    _interpreter = await Interpreter.fromAsset('assets/best_float32.tflite');
    final rawLabels = await rootBundle.loadString('assets/label.txt');
    _labels =
        rawLabels
            .split('\n')
            .map((l) => l.trim())
            .where((l) => l.isNotEmpty)
            .toList();
  }

  void _initFaceDetector() {
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        performanceMode: FaceDetectorMode.accurate,
        enableLandmarks: false,
        enableContours: false,
      ),
    );
  }

  Future<void> _pickImage(ImageSource src) async {
    final picked = await ImagePicker().pickImage(source: src);
    if (picked == null) return;

    setState(() {
      _selectedImage = File(picked.path);
      _selectedVideo = null;
      _faceBoxes = [];
      _predictions = [];
      _processedFrames.clear();
      _frameLabels.clear();
    });

    await _detectAndClassifyImage(_selectedImage!);
  }

  Future<void> _detectAndClassifyImage(File file) async {
    final inputImg = InputImage.fromFile(file);
    final faces = await _faceDetector.processImage(inputImg);

    if (faces.isEmpty) {
      setState(() {
        _predictions = ['Tidak ada wajah terdeteksi.'];
      });
      return;
    }

    final bytes = await file.readAsBytes();
    _originalImage = img.decodeImage(bytes);

    final boxes = <Rect>[];
    final preds = <String>[];

    for (final face in faces) {
      final box = face.boundingBox;
      final crop = img.copyCrop(
        _originalImage!,
        x: box.left.toInt().clamp(0, _originalImage!.width - 1),
        y: box.top.toInt().clamp(0, _originalImage!.height - 1),
        width: box.width.toInt().clamp(1, _originalImage!.width),
        height: box.height.toInt().clamp(1, _originalImage!.height),
      );
      final resized = img.copyResize(crop, width: 224, height: 224);
      final input = _imageToInput(resized);

      var output = List.filled(
        _labels.length,
        0.0,
      ).reshape([1, _labels.length]);
      _interpreter.run([input], output);

      final probs = (output[0] as List).cast<double>();
      final max = probs.reduce((a, b) => a > b ? a : b);
      final idx = probs.indexOf(max);

      boxes.add(box);
      preds.add('${_labels[idx]} (${(max * 100).toStringAsFixed(1)}%)');
    }

    setState(() {
      _faceBoxes = boxes;
      _predictions = preds;
    });
  }

  List<List<List<double>>> _imageToInput(img.Image image) {
    return List.generate(224, (y) {
      return List.generate(224, (x) {
        final px = image.getPixel(x, y);
        return [px.r / 255.0, px.g / 255.0, px.b / 255.0];
      });
    });
  }

  Future<void> _pickVideo() async {
    final picked = await ImagePicker().pickVideo(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() {
      _selectedVideo = File(picked.path);
      _selectedImage = null;
      _faceBoxes = [];
      _predictions = [];
      _processedFrames.clear();
      _frameLabels.clear();
    });

    await _processVideo(_selectedVideo!);
  }

  Future<void> _processVideo(File videoFile) async {
    setState(() {
      _isProcessing = true;
      _processedFrames.clear();
      _frameLabels.clear();
    });

    final controller = VideoPlayerController.file(videoFile);
    await controller.initialize();
    final totalSeconds = controller.value.duration.inSeconds;
    await controller.dispose();

    final maxSeconds = totalSeconds > 60 ? 60 : totalSeconds;
    for (int sec = 0; sec < maxSeconds; sec++) {
      final bytes = await VideoThumbnail.thumbnailData(
        video: videoFile.path,
        timeMs: sec * 1000,
        imageFormat: ImageFormat.JPEG,
        quality: 80,
      );
      if (bytes == null) continue;

      final frameImg = img.decodeImage(bytes);
      if (frameImg == null) continue;

      await _classifyFrame(frameImg, addToList: true);
    }

    setState(() => _isProcessing = false);
  }

  Future<String> _classifyFrame(
    img.Image frame, {
    bool addToList = false,
  }) async {
    final tmp = await _saveTempJpeg(frame);
    final faces = await _faceDetector.processImage(
      InputImage.fromFilePath(tmp.path),
    );
    tmp.deleteSync();

    String label;
    if (faces.isEmpty) {
      label = 'No face';
    } else {
      final box = faces.first.boundingBox;
      final crop = img.copyCrop(
        frame,
        x: box.left.toInt().clamp(0, frame.width - 1),
        y: box.top.toInt().clamp(0, frame.height - 1),
        width: box.width.toInt().clamp(1, frame.width),
        height: box.height.toInt().clamp(1, frame.height),
      );
      final resized = img.copyResize(crop, width: 224, height: 224);
      final input = _imageToInput(resized);

      var output = List.filled(
        _labels.length,
        0.0,
      ).reshape([1, _labels.length]);
      _interpreter.run([input], output);

      final probs = (output[0] as List).cast<double>();
      final max = probs.reduce((a, b) => a > b ? a : b);
      final idx = probs.indexOf(max);
      label = '${_labels[idx]} (${(max * 100).toStringAsFixed(1)}%)';
    }

    if (addToList) {
      _processedFrames.add(frame);
      _frameLabels.add(label);
    }

    return label;
  }

  Future<File> _saveTempJpeg(img.Image image) async {
    final jpg = img.encodeJpg(image);
    final dir = await getTemporaryDirectory();
    final file = File(
      '${dir.path}/${DateTime.now().millisecondsSinceEpoch}.jpg',
    );
    await file.writeAsBytes(jpg);
    return file;
  }

  // Permintaan permission
  Future<bool> _requestPermission() async {
    if (Platform.isAndroid) {
      // Android < 11: STORAGE
      if (await Permission.storage.isGranted) return true;

      // Android 11–12: MANAGE_EXTERNAL_STORAGE (all files)
      if (Platform.isAndroid &&
          (await Permission.manageExternalStorage.isDenied)) {
        final status = await Permission.manageExternalStorage.request();
        return status.isGranted;
      }

      // Android < 11: fallback ke storage
      final status = await Permission.storage.request();
      return status.isGranted;
    } else if (Platform.isIOS) {
      // iOS: photo library
      final status = await Permission.photos.request();
      return status.isGranted;
    }
    // other platforms
    return true;
  }
  
  // Simpan ke galeri
  Future<void> _saveProcessedImage() async {
    if (_selectedImage == null || _originalImage == null) return;
    if (!await _requestPermission()) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Storage permission denied')),
      );
      return;
    }

    // 1. Render gambar dengan kotak & label
    final painted = await _renderImageWithBoxes();
    // 2. Encode ke PNG bytes
    final pngBytes = img.encodePng(painted);
    final bytes = Uint8List.fromList(pngBytes);

    // 3. Simpan ke galeri
    final result = await ImageGallerySaverPlus.saveImage(
      bytes,
      quality: 100, // opsi kualitas
      name: 'emotion_${DateTime.now().millisecondsSinceEpoch}',
    );

    // 4. Tampilkan notifikasi
    final success = (result == true || result['isSuccess'] == true);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          success ? 'Gambar tersimpan di galeri!' : 'Gagal menyimpan gambar',
        ),
      ),
    );
  }

  Future<img.Image> _renderImageWithBoxes() async {
    final image = img.copyResize(_originalImage!, width: 800);
    final scaleX = image.width / _originalImage!.width;
    final scaleY = image.height / _originalImage!.height;

    for (int i = 0; i < _faceBoxes.length; i++) {
      final box = _faceBoxes[i];
      img.drawRect(
        image,
        x1: (box.left * scaleX).toInt(),
        y1: (box.top * scaleY).toInt(),
        x2: (box.right * scaleX).toInt(),
        y2: (box.bottom * scaleY).toInt(),
        color: img.ColorRgba8(255, 0, 0, 255),
        thickness: 3,
      );
      img.drawString(
        image,
        _predictions[i],
        font: img.arial24,
        x: (box.left * scaleX).toInt(),
        y: (box.bottom * scaleY + 4).toInt(),
        color: img.ColorRgba8(255, 255, 255, 255),
      );
    }
    return image;
  }

  void _reset() {
    setState(() {
      _selectedImage = null;
      _selectedVideo = null;
      _originalImage = null;
      _faceBoxes = [];
      _predictions = [];
      _isProcessing = false;
      _processedFrames.clear();
      _frameLabels.clear();
    });
  }

  @override
  void dispose() {
    _faceDetector.close();
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Emotion Detector'),
        backgroundColor: Colors.deepOrangeAccent,
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Container(
                width: double.infinity,
                color: Colors.grey[900],
                child:
                    (_selectedImage != null && _originalImage != null)
                        ? LayoutBuilder(
                          builder: (ctx, constraints) {
                            final ar =
                                _originalImage!.width / _originalImage!.height;
                            final width = constraints.maxWidth;
                            final height = width / ar;
                            return SizedBox(
                              width: width,
                              height: height,
                              child: Stack(
                                fit: StackFit.expand,
                                children: [
                                  Image.file(
                                    _selectedImage!,
                                    fit: BoxFit.cover,
                                  ),
                                  CustomPaint(
                                    painter: FaceBoxPainter(
                                      faceRects: _faceBoxes,
                                      labels: _predictions,
                                      originalSize: Size(
                                        _originalImage!.width.toDouble(),
                                        _originalImage!.height.toDouble(),
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            );
                          },
                        )
                        : Center(
                          child: Text(
                            'Pilih Gambar atau Video',
                            style: TextStyle(color: Colors.white54),
                          ),
                        ),
              ),
            ),
            const SizedBox(height: 12),
            if (_processedFrames.isNotEmpty)
              SizedBox(
                height: 140,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: _processedFrames.length,
                  itemBuilder: (context, index) {
                    final bytes = Uint8List.fromList(
                      img.encodeJpg(_processedFrames[index]),
                    );
                    return Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 4),
                      child: Column(
                        children: [
                          Image.memory(bytes, height: 100),
                          const SizedBox(height: 4),
                          Text(
                            _frameLabels[index],
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            const SizedBox(height: 12),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  Expanded(
                    child: OptionButton(
                      icon: Icons.photo,
                      label: 'Gambar',
                      onTap: () => _pickImage(ImageSource.gallery),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: OptionButton(
                      icon: Icons.camera_alt,
                      label: 'Kamera',
                      onTap: () => _pickImage(ImageSource.camera),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: OptionButton(
                      icon: Icons.video_library,
                      label: 'Video',
                      onTap: _pickVideo,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  if (_selectedImage != null)
                    Expanded(
                      child: OptionButton(
                        icon: Icons.download,
                        label: 'Simpan',
                        onTap: _saveProcessedImage,
                      ),
                    ),
                  if (_selectedImage != null || _selectedVideo != null)
                    const SizedBox(width: 8),
                  if (_selectedImage != null || _selectedVideo != null)
                    Expanded(
                      child: OptionButton(
                        icon: Icons.refresh,
                        label: 'Reset',
                        onTap: _reset,
                      ),
                    ),
                ],
              ),
            ),
            const SizedBox(height: 16),
          ],
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
    Key? key,
    required this.icon,
    required this.label,
    required this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Ink(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF9C4521), Color(0xFFD98248)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: Colors.white),
            const SizedBox(width: 4),
            Flexible(
              child: Text(
                label,
                style: const TextStyle(color: Colors.white),
                overflow: TextOverflow.ellipsis,
                maxLines: 1,
              ),
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
    final paint =
        Paint()
          ..color = Colors.red
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3;

    final scaleX = size.width / originalSize.width;
    final scaleY = size.height / originalSize.height;

    for (int i = 0; i < faceRects.length; i++) {
      final r = faceRects[i];
      final sr = Rect.fromLTRB(
        r.left * scaleX,
        r.top * scaleY,
        r.right * scaleX,
        r.bottom * scaleY,
      );
      canvas.drawRect(sr, paint);

      final span = TextSpan(
        text: labels[i],
        style: const TextStyle(color: Colors.white, fontSize: 14),
      );
      final tp = TextPainter(text: span, textDirection: TextDirection.ltr);
      tp.layout();
      tp.paint(canvas, Offset(sr.left, sr.bottom + 4));
    }
  }

  @override
  bool shouldRepaint(covariant FaceBoxPainter oldDelegate) => true;
}
