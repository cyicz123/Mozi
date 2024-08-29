import 'dart:developer';

import 'package:flutter/material.dart';
import 'package:window_manager/window_manager.dart';
import 'app_bar.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 确保在桌面平台上运行
  await windowManager.ensureInitialized();

  WindowOptions windowOptions = const WindowOptions(
    minimumSize: Size(400, 300),
    size: Size(800, 600),
    center: true,
    backgroundColor: Colors.transparent,
    skipTaskbar: false,
    titleBarStyle: TitleBarStyle.normal,
  );

  windowManager.waitUntilReadyToShow(windowOptions, () async {
    await windowManager.show();
    await windowManager.focus();
  });
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  ThemeMode _themeMode = ThemeMode.system;
  final seedColor = const Color.fromARGB(255, 157, 32, 32);

  bool get useLightTheme => switch (_themeMode) {
        ThemeMode.system =>
          View.of(context).platformDispatcher.platformBrightness ==
              Brightness.light,
        ThemeMode.light => true,
        ThemeMode.dark => false
      };
  
  void handleThemeChanged(bool useLightTheme) {
    setState(() {
      _themeMode = useLightTheme ? ThemeMode.light : ThemeMode.dark;
    });
  }

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      themeMode: _themeMode,
      theme: ThemeData(
        colorSchemeSeed: seedColor,
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: seedColor,
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: MyHomePage(
        handleBrightnessChange: handleThemeChanged,
      ),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({
    super.key,
    required this.handleBrightnessChange,
  });
  final void Function(bool uselightMode) handleBrightnessChange;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    final isBright = Theme.of(context).brightness == Brightness.light;

    return Scaffold(
      body: Row(children: [
        VerticalAppBar(
          onHomePressed: () {
            // 实现主页按钮逻辑
            log('主页按钮被点击');
          },
          onThemeToggled: () => widget.handleBrightnessChange(!isBright),
          onAboutPressed: () {
            // 实现关于按钮逻辑
            log('关于按钮被点击');
          },
        ),
        Expanded(
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                const Text(
                  'You have pushed the button this many times:',
                ),
                Text(
                  '$_counter',
                  style: Theme.of(context).textTheme.headlineMedium,
                ),
              ],
            ),
          ),
        ),
      ]),
      floatingActionButton: FloatingActionButton(
        onPressed: _incrementCounter,
        tooltip: 'Increment',
        child: const Icon(Icons.add),
      ), // This trailing comma makes auto-formatting nicer for build methods.
    );
  }
}
