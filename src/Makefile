VERSION = 1.0
CC = clang++
CPPFLAGS = -O3 -ftree-vectorize -funroll-loops -fomit-frame-pointer -ffast-math -Wall

DEPS =	main.h \
		Image.h \
		GrayscaleImage.h \
		RGBImage.h \
		RGBAImage.h \
		HSIImage.h \
		YCrCbImage.h \
		EdgeDetector.h \
		LinearFilter.h \
		MorphologicalFilter.h \
		TypeConversion.h \
		Parameter.h \
		Util.h \
		HuberROFModel.h \
		ImageInpainting.h \
		RealTimeMinimizer.h \
		TVL1Model.h

OBJ =	main.o \
		Image.o \
		GrayscaleImage.o \
		RGBImage.o \
		RGBAImage.o \
		HSIImage.o \
		YCrCbImage.o \
		EdgeDetector.o \
		LinearFilter.o \
		MorphologicalFilter.o \
		TypeConversion.o \
		Parameter.o \
		Util.o \
		HuberROFModel.o \
		ImageInpainting.o \
		RealTimeMinimizer.o \
		TVL1Model.o

LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc
LDFLAGS = -L/opt/local/lib
IDFLAGS = -I/opt/local/include

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CPPFLAGS) $(IDFLAGS)

iPaur: $(OBJ)
	$(CC) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS)

# clean:
# 	rm *.o