This is a c++ implementation of the K-Means clustering algorithm. It also includes the KMeans++ initialization algorithm.

Obtaining the code:
Please be sure to:
 git clone --recursive 
 
so that the submodules will also be checked out.

Example usage:

  KMeansClustering::VectorOfPoints points = ... load data ...
  KMeansClustering kmeans;
  kmeans.SetK(2); // specify the number of clusters
  kmeans.SetPoints(points);
  kmeans.Cluster(); // Perform the clustering

  // Get the membership of every point
  std::vector<unsigned int> labels = GetLabels();

  // Get all points ids with a particular membership (in this case, cluster 0)
  std::vector<unsigned int> pointIdsInCluster0 = kmeans.GetIndicesWithLabel(0);

  /** Get the points with a specified cluster membership. */
  VectorOfPoints pointsInCluster0 = kmeans.GetPointsWithLabel(0);
