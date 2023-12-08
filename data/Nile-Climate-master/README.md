# Nile-Climate
Analysis of change-points in the nilometer low-flow data record

Nile-Climate is a quick analysis using the Lanzante 1996 change-point detection method to find significant change-points in the low flow record of the Nile River between 622-1284 recorded using a nilometer (a staircase with descending into the river from which priests measured the lowest river level during the dry season). 

Method: Nile data is read into niledata and is normalized by subtracting the median and dividing by the standard deviation. The Lanzante 1996 method is used to determine changepoints in the data record (see paper for more detail).

Note the 715 changepoint is reportedly due to a redesign of the apparatus

Nilometer data is taken from http://mldata.org/repository/data/viewslug/nile-water-level/ on June 27 2016

Lanzante, J. R (1996) Resistant, robust and non-parametric technique for the analysis of climate data: theory and examples including applications to historical radiosonde station data. International Journal of Climatology. 16, 1197-1226.
