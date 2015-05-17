# ImageProcessing: main.cpp image.cpp edge_detection.cpp
# 	g++ -L/usr/local/lib -I/usr/local/include -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lm main.cpp image.cpp edge_detection.cpp -o ImageProcessing

ImageProcessing: main.cpp image.cpp ms_minimizer.cpp
	g++ -O3 -ftree-vectorize -funroll-loops -fomit-frame-pointer -ffast-math -L/usr/local/lib -I/usr/local/include/libpng12 -lpng -fopenmp main.cpp image.cpp ms_minimizer.cpp -o ImageProcessing


# VERSION = 1.00
# CXX = g++
# CC = gcc
# CFLAGS = -O2 -Wall -std=c99 -fopenmp
# CPPFLAGS = -O2 -Wall
# DEPS = image.h ms_minimizer.h
# OBJ = main.o image.o ms_minimizer.o
# LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lm
# LDFLAGS = -L/usr/local/lib
# IDFLAGS = -I/usr/local/include

# %.o: %.c $(DEPS)
# 	$(CC) -c -o $@ $< $(CFLAGS)

# %.o: %.cpp $(DEPS)
# 	$(CXX) -c -o $@ $< $(CPPFLAGS) $(IDFLAGS)

# ImageProcessing: $(OBJ)
# 	$(CC) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

# clean:
# 	rm *.o