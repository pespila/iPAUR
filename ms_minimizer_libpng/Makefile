# ImageProcessing: main.cpp image.cpp edge_detection.cpp
# 	g++ -L/usr/local/lib -I/usr/local/include -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lm main.cpp image.cpp edge_detection.cpp -o ImageProcessing

# ImageProcessing: main.cpp image.cpp util.cpp huber_rof_model.cpp
# 	g++ -O3 -ftree-vectorize -funroll-loops -fomit-frame-pointer -ffast-math -L/usr/local/lib -I/usr/local/include/libpng12 -lpng -fopenmp main.cpp image.cpp util.cpp huber_rof_model.cpp -o ImageProcessing


VERSION = 1.00
CC = g++
CFLAGS = -O3 -ftree-vectorize -funroll-loops -fomit-frame-pointer -ffast-math -Wall
DEPS = image.h util.h huber_rof_model.h tv_l1_model.h
OBJ = main.o image.o util.o huber_rof_model.o tv_l1_model.o
LDLIBS = -lpng -fopenmp
LDFLAGS = -L/usr/local/lib
IDFLAGS = -I/usr/local/include/libpng12

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(IDFLAGS)

ImageProcessing: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LDLIBS)

clean:
	rm *.o