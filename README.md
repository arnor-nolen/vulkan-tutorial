## Vulkan tutorial

Simple rendering engine using Vulkan.

## Dependencies

Vulkan SDK has to be installed in the system. For the rest of dependencies, I use `conan` package manager to pull them all. This project's dependencies can be seen in [conanfile.txt](./conanfile.txt).

## Compilation info

To compile the project, first you'll need to install `conan` to manage the dependencies:

```
pip install conan
```

If you don't have `pip` installed, refer to [this](https://docs.conan.io/en/latest/installation.html) `conan` installation guide.

Next, we're creating two profiles for Debug and Release:

```
cd build
conan install .. -s build_type=Debug -if Debug
conan install .. -s build_type=Release -if Release
```

After that, the easiest way to build the application is by using VS Code [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) extension.

## Building manually

If you're using other editor or don't want to use the CMake Tools extension, you'll have to build the project manually.
First, use CMake to generate the appropriate build files (replace **Release** with **Debug** if needed):

```
cd Release
cmake ../.. -DCMAKE_BUILD_TYPE=RELEASE
```

Using generated files, you can compile the project. On OSX/Linux use:

```
cmake --build .
```

On Windows, you have to specify the build type:

```
cmake --build . --config RELEASE
```

Now, enjoy your freshly minted binaries inside the **bin** folder!

## Cleaning up build files

If you want to clean up the build files and binaries, you can use `git` from the project root directory:

```
git clean -dfX build
```
