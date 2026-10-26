CXX := g++
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -pedantic -Iinclude
LDFLAGS :=

BUILD_DIR := build
TARGET := $(BUILD_DIR)/mp48mb
PORTABLE_TARGET := $(BUILD_DIR)/mp48mb-portable.exe
TEST_TARGET := $(BUILD_DIR)/test_quadtree

HEADERS := include/codec.hpp include/frame.hpp include/quadtree_stack.hpp
SRCS := src/main.cpp src/codec.cpp
TEST_SRCS := src/test_quadtree.cpp src/codec.cpp

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	@if not exist "$(BUILD_DIR)" mkdir "$(BUILD_DIR)"
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(PORTABLE_TARGET): $(SRCS) $(HEADERS) Makefile
	@if not exist "$(BUILD_DIR)" mkdir "$(BUILD_DIR)"
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS) -static -static-libgcc -static-libstdc++

package-backend: $(PORTABLE_TARGET)

$(TEST_TARGET): $(TEST_SRCS) $(HEADERS)
	@if not exist "$(BUILD_DIR)" mkdir "$(BUILD_DIR)"
	$(CXX) $(CXXFLAGS) $(TEST_SRCS) -o $@ $(LDFLAGS)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

benchmark: $(TARGET)
	./$(TARGET) benchmark 640 480

gui: $(TARGET)
	python src/desktop_workbench.py

test-python:
	python -m unittest discover -s src -p "test_*.py"

clean:
	@if exist "$(BUILD_DIR)" rmdir /s /q "$(BUILD_DIR)"

.PHONY: all package-backend test test-python benchmark gui clean
