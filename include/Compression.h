#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <map>

class Compression {
private:
  Eigen::VectorXd Signal, Coefficients, ThresholdedCoefficients;
  Eigen::MatrixXd Wavelet, Inverse;
  int size_wavelet = 8;
  std::map<int, float> nonZeroValues;
  int size_signal = 0;
  double threshold = 0.5;

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