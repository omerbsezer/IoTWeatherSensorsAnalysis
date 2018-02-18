# Internet of Things (IoT) Weather Sensors Analysis

"In this study an extended IoT Framework that integrates the data retrieval, processing, and learning layers is presented with a use case on weather data clustering analysis. The learning model we developed uses clustering unsupervised learning method in the learning phase of the framework in order to best utilize the associated big data for this problem. The US Weather data captured from 8000 different weather stations around North America is acquired through log files. In this particular study, air temperature, wind-speed, relative humidity, visibility, and pressure data are used in the data analysis. Traditional k-means clustering algorithm is applied and the results are presented. As an interesting phenomena, we observed that the data clustering matches the geographical alignment of the stations. In other words, some of the important geographical regions within the North American continent (and the continental USA) form distinct weather clusters and easily differentiated from each other. In addition, possible sensor faults and anomalies are emerged with using clustering method. This use case allowed us to present an example of how such a IoT Big Data framework can be used for such implementations."

In this study, proposed "An Extended IoT Framework" learning part is presented with a use case on weather data clustering analysis.

LinkedSensorData and LinkedObservationData are used for use case scenario. 

Link: [http://wiki.knoesis.org/index.php/LinkedSensorData](http://wiki.knoesis.org/index.php/LinkedSensorData)

"LinkedSensorData is an RDF dataset containing expressive descriptions of ~20,000 weather stations in the United States."
"LinkedObservationData is an RDF dataset containing expressive descriptions of hurricane and blizzard observations in the United States."

They are converted from RDF to CSV file to process with ML algorithms.

Input Data Set: /SensorsDataSet

Output: /Results (that plots clusters and cluster detail information)

IEEE Link: http://ieeexplore.ieee.org/document/8258150/

ResearchGate Link: https://www.researchgate.net/publication/322515935_Weather_data_analysis_and_sensor_fault_detection_using_an_extended_IoT_framework_with_semantics_big_data_and_machine_learning

_**Cite as:**_

**Bibtex:**

```
@INPROCEEDINGS{8258150,
  author={A. C. Onal and O. Berat Sezer and M. Ozbayoglu and E. Dogdu},
  booktitle={2017 IEEE International Conference on Big Data (Big Data)},
  title={Weather data analysis and sensor fault detection using an extended IoT framework with semantics, big data, and machine learning},
  year={2017},
  volume={},
  number={},
  pages={2037-2046},
  keywords={Big Data;Data analysis;Feature extraction;Machine learning algorithms;Meteorology;Resource description framework;Semantics;Internet of things;anomaly detection;big data analytics;clustering;fault detection;framework;machine learning;weather data analysis},
  doi={10.1109/BigData.2017.8258150},
  ISSN={},
  month={Dec},
}
```

What is K-Means Clustering? (General Information): https://en.wikipedia.org/wiki/K-means_clustering

Scikit-learn: http://scikit-learn.org/stable/
