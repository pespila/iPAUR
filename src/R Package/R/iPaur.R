library(png)

ROFmodel <- function(src.file, dst.file, lambda, tau, iter, src.path = "../img/", dst.path = "../") {
  img.path <- paste(src.path, src.file, sep="")
  approx.path <- paste(dst.path, dst.file, sep="")
  img <- readPNG(source = img.path)
  img <- img * 255
  approx <- img
  ROF(img, approx, lambda, tau, iter, dim(img)[2], dim(img)[1], 1)
  approx <- approx / 255
  writePNG(image = approx, target = approx.path)
}

TVL1model <- function(src.file, dst.file, lambda, tau, iter, src.path = "../img/", dst.path = "../") {
  img.path <- paste(src.path, src.file, sep="")
  approx.path <- paste(dst.path, dst.file, sep="")
  img <- readPNG(source = img.path)
  img <- img * 255
  approx <- img
  TVL1(img, approx, lambda, tau, iter, dim(img)[2], dim(img)[1], 1)
  approx <- approx / 255
  writePNG(image = approx, target = approx.path)
}

HuberROFmodel <- function(src.file, dst.file, alpha, lambda, tau, iter, src.path = "../img/", dst.path = "../") {
  img.path <- paste(src.path, src.file, sep="")
  approx.path <- paste(dst.path, dst.file, sep="")
  img <- readPNG(source = img.path)
  img <- img * 255
  approx <- img
  HuberROF(img, approx, alpha, lambda, tau, iter, dim(img)[2], dim(img)[1], 1)
  approx <- approx / 255
  writePNG(image = approx, target = approx.path)
}

Inpainting <- function(src.file, dst.file, lambda, tau, iter, src.path = "../img/", dst.path = "../") {
  img.path <- paste(src.path, src.file, sep="")
  approx.path <- paste(dst.path, dst.file, sep="")
  img <- readPNG(source = img.path)
  img <- img * 255
  approx <- img
  Inpaint(img, approx, lambda, tau, iter, dim(img)[2], dim(img)[1], 1)
  approx <- approx / 255
  writePNG(image = approx, target = approx.path)
}

RealTimeMinimizer <- function(src.file, dst.file, alpha, lambda, tau, cartoon, iter, src.path = "../img/", dst.path = "../") {
  img.path <- paste(src.path, src.file, sep="")
  approx.path <- paste(dst.path, dst.file, sep="")
  img <- readPNG(source = img.path)
  approx <- img
  RealTime(img, approx, alpha, lambda, tau, cartoon, iter, dim(img)[2], dim(img)[1], 1)
  writePNG(image = approx, target = approx.path)
}

Sobel <- function(src.file, dst.file, src.path = "../img/", dst.path = "../") {
  img.path <- paste(src.path, src.file, sep="")
  approx.path <- paste(dst.path, dst.file, sep="")
  img <- readPNG(source = img.path)
  img <- img * 255
  approx <- img
  SobelOperator(img, approx, dim(img)[2], dim(img)[1], 1)
  approx <- approx / 255
  writePNG(image = approx, target = approx.path)
}