VERSION = 1.00
CC = g++
CPPFLAGS = -O2 -Wall
DEPS = main.h image.h ms_minimizer.h analytical_operators.h
OBJ = main.o image.o ms_minimizer.o analytical_operators.o
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -lm
LDFLAGS = -L/usr/local/lib
IDFLAGS = -I/usr/local/include

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CPPFLAGS) $(IDFLAGS)

ImageProcessing: $(OBJ)
	$(CC) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

clean:
	rm *.o