VERSION = 1.00
CC = g++
CPPFLAGS = -I/usr/local/lib/gcc/x86_64-apple-darwin14.0.0/5.0.0/include -O2 -Wall -fopenmp
DEPS = main.h image.h ms_minimizer.h ms_minimizer_video.h analytical_operators.h
OBJ = main.o image.o ms_minimizer.o ms_minimizer_video.o analytical_operators.o
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lm
LDFLAGS = -L/usr/local/lib
IDFLAGS = -I/usr/local/include

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CPPFLAGS) $(IDFLAGS)

ImageProcessing: $(OBJ)
	$(CC) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

clean:
	rm *.o