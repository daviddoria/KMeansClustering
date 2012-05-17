/*=========================================================================
 *
 *  Copyright David Doria 2011 daviddoria@gmail.com
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

/*
KMeans clustering is a method in which to form K (known) clusters of points from
an unorganized set of input points.
*/

#ifndef KMeansClustering_h
#define KMeansClustering_h

// STL
#include <vector>

// Eigen
#include <Eigen/Dense>

class KMeansClustering
{
public:
  typedef std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf> > VectorOfPoints;

  /** Constructor. */
  KMeansClustering();

  /** The number of clusters to find */
  void SetK(const unsigned int k);
  unsigned int GetK();

  std::vector<unsigned int> GetIndicesWithLabel(const unsigned int label);
  VectorOfPoints GetPointsWithLabel(const unsigned int label);

  /**
   * If this function is called, the randomness
   * is removed for repeatability for testing
   */
  void SetRandom(const bool r);

  /** Set the points to cluster. */
  void SetPoints(const VectorOfPoints& points);

  std::vector<unsigned int> GetLabels();
  
  /** Set which initialization method to use. */
  void SetInitMethod(const int method);

  /** Choices of initialization methods */
  enum InitMethodEnum{RANDOM, KMEANSPP};

  /** Actually perform the clustering. */
  void Cluster();

  void OutputClusterCenters();
  
protected:

  /** Randomly initialize cluster centers */
  void RandomInit();

  /** Initialize cluster centers using the KMeans++ algorithm */
  void KMeansPPInit();

  unsigned int ClosestCluster(const Eigen::VectorXf& queryPoint);
  
  unsigned int ClosestPointIndex(const Eigen::VectorXf& queryPoint);
  double ClosestPointDistance(const Eigen::VectorXf& queryPoint);
  double ClosestPointDistanceExcludingId(const Eigen::VectorXf& queryPoint, const unsigned int excludedId);
  double ClosestPointDistanceExcludingIds(const Eigen::VectorXf& queryPoint, const std::vector<unsigned int> excludedIds);

  /** Based on the current cluster membership, compute the cluster centers. */
  void EstimateClusterCenters();
  
  void AssignLabels();
  bool CheckChanged(const std::vector<unsigned int> labels, const std::vector<unsigned int> oldLabels);

  Eigen::VectorXf GetRandomPointInBounds();
  unsigned int SelectWeightedIndex(std::vector<double> weights); // Intentionally not passed by reference

private:

  /** The label (cluster ID) of each point. */
  std::vector<unsigned int> Labels;

  /** Should the computation be random? If false, then it is repeatable (for testing). */
  bool Random;

  /** The initialization method to use */
  int InitMethod;

  /** The number of clusters to find */
  unsigned int K;
  
  /** The points to cluster. */
  VectorOfPoints Points;

  /** The current cluster centers. */
  VectorOfPoints ClusterCenters;
};

#endif
