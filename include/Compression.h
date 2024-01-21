#include <Eigen/Dense>
#include <iostream>

class Compression {
private:
  Eigen::VectorXd Signal, Coefficients, ThresholdedCoefficients;
  Eigen::MatrixXd Wavelet, Inverse;
  int size_wavelet;
  int size_signal;
  double threshold = 4;

public:
  Compression(int signalSize);
  ~Compression();
  void haarWaveletMatrix();
  void findCoefficients();
  void applyThreshold();
  void regenerateSignal();
  double calculateMSE();
  double calculatePSNR();
  void printCoefficients();
  void printThresholdedCoefficients();
  void printSignal();
  void printCompressionRatio();
  const Eigen::VectorXd &getSignal() const;
};