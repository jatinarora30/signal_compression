#include "../include/Compression.h"
Compression::Compression(int signalSize)
    : size_signal(signalSize), size_wavelet(signalSize) {
  std::cout << "[Compression] Compression" << std::endl;
  Signal = Eigen::VectorXd::Random(size_signal);

  Coefficients.resize(size_signal);
  Coefficients.setOnes();

  Wavelet.resize(size_wavelet, size_wavelet);
  haarWaveletMatrix();
  Inverse = Wavelet.inverse();
}

void Compression::printCompressionRatio() {
  std::cout << "Original Signal Size: " << size_signal << " elements"
            << std::endl;
  std::cout << "Compressed Signal Size: " << nonZeroValues.size()
            << " elements" << std::endl;
  double compressionRatio =
      static_cast<double>(size_signal) / nonZeroValues.size();
  std::cout << "Compression Ratio: " << compressionRatio << std::endl;
}

Compression::~Compression() {
  std::cout << "[Compression] ~Compression" << std::endl;
}

void Compression::haarWaveletMatrix() {
  int currentSize = size_wavelet;
  Wavelet.resize(size_wavelet, size_wavelet);

  while (currentSize > 1) {
    currentSize /= 2;
    Eigen::MatrixXd subMatrix(currentSize * 2, currentSize * 2);

    // Build the lower-left block
    for (int i = 0; i < currentSize * 2; ++i) {
      if (i < currentSize)
        subMatrix(i, i) = 1;

      if (i + currentSize < currentSize * 2)
        subMatrix(i, i + currentSize) = 1;
    }

    // Build the upper-right block
    for (int i = 0; i < currentSize * 2; ++i) {
      if (i < currentSize)
        subMatrix(i + currentSize, i) = 1;

      if (i + currentSize < currentSize * 2)
        subMatrix(i + currentSize, i + currentSize) = -1;
    }

    Wavelet.block(0, 0, currentSize * 2, currentSize * 2) = subMatrix;
  }
}

void Compression::printSignal() {
  std::cout << "Generated Signal:\n" << Signal.transpose() << std::endl;
}

void Compression::findCoefficients() {
  Coefficients = Inverse * Signal;
  applyThreshold();

  // Print non-zero indices and values
  std::cout << "Non-Zero Indices and Values: ";
  for (const auto &entry : nonZeroValues) {
    std::cout << entry.first << ":" << entry.second << " ";
  }
  std::cout << std::endl;

  // Create ThresholdedCoefficients using nonZeroValues
  ThresholdedCoefficients = Eigen::VectorXd::Zero(size_signal);
  for (const auto &entry : nonZeroValues) {
    int index = entry.first;
    float value = entry.second;
    ThresholdedCoefficients[index] = value;
  }
}

void Compression::applyThreshold() {
  nonZeroValues.clear(); // Clear previous non-zero values
  for (int i = 0; i < Coefficients.size(); ++i) {
    float value = Coefficients[i];
    if (std::abs(value) > threshold) {
      nonZeroValues[i] = value;
    }
  }
}

void Compression::regenerateSignal() {
  Signal = Eigen::VectorXd::Zero(size_signal);

  for (const auto &entry : nonZeroValues) {
    int index = entry.first;
    float value = entry.second;
    Signal += Inverse.col(index) * value;
  }
}

double Compression::calculateMSE() {
  Eigen::VectorXd error = Signal - ThresholdedCoefficients;
  return error.squaredNorm() / size_signal;
}

double Compression::calculatePSNR() {
  double mse = calculateMSE();
  double maxSignalValue = Signal.maxCoeff();
  return 20 * log10(maxSignalValue) - 10 * log10(mse);
}

void Compression::printCoefficients() {
  std::cout << "Coefficients:\n" << Coefficients.transpose() << std::endl;
}

void Compression::printThresholdedCoefficients() {
  std::cout << "Thresholded Coefficients:\n"
            << ThresholdedCoefficients.transpose() << std::endl;
}

const Eigen::VectorXd &Compression::getSignal() const { return Signal; }

int main() {
  int signalSize = 8; // Change this to your desired signal size
  Compression comp(signalSize);

  comp.printSignal();
  comp.findCoefficients();

  comp.printCoefficients();
  comp.printThresholdedCoefficients();

  comp.regenerateSignal();
  std::cout << "Regenerated Signal:\n"
            << comp.getSignal().transpose() << std::endl;

  double mse = comp.calculateMSE();
  double psnr = comp.calculatePSNR();

  std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
  std::cout << "Peak Signal-to-Noise Ratio (PSNR): " << psnr << " dB"
            << std::endl;
  comp.printCompressionRatio();

  return 0;
}