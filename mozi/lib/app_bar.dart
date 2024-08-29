import 'package:flutter/material.dart';

class VerticalAppBar extends StatelessWidget {
  final VoidCallback onHomePressed;
  final VoidCallback onThemeToggled;
  final VoidCallback onAboutPressed;

  const VerticalAppBar({
    super.key,
    required this.onHomePressed,
    required this.onThemeToggled,
    required this.onAboutPressed,
  });

  @override
  Widget build(BuildContext context) {
    final themeColor = Theme.of(context).colorScheme;
    final isBright = Theme.of(context).brightness == Brightness.light;

    return Container(
      width: 60, // 调整宽度以适应您的需求
      color: themeColor.primary,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          const SizedBox(height: 20),
          IconButton(
            icon: Icon(Icons.home, color: themeColor.onPrimary),
            onPressed: onHomePressed,
          ),
          const Spacer(),
          IconButton(
            icon: Icon(
              isBright ? Icons.dark_mode : Icons.light_mode,
              color: themeColor.onPrimary,
            ),
            onPressed: onThemeToggled,
          ),
          const SizedBox(height: 20),
          IconButton(
            icon: Icon(Icons.info_outline, color: themeColor.onPrimary),
            onPressed: onAboutPressed,
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }
}
