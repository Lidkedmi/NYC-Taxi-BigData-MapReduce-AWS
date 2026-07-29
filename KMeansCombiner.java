package com.taxi.kmeans;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * KMeans Combiner - a local optimization that runs on each mapper node
 * before the shuffle phase. It partially sums the points belonging to the
 * same centroid, reducing the amount of data sent to reducers.
 * The summed Point keeps a count so the reducer can still compute the mean.
 */
public class KMeansCombiner extends Reducer<IntWritable, Point, IntWritable, Point> {

    @Override
    protected void reduce(IntWritable key, Iterable<Point> values, Context context)
            throws IOException, InterruptedException {

        Point partialSum = null;
        for (Point p : values) {
            if (partialSum == null) {
                partialSum = new Point(p.getCoords().clone());
                partialSum.setCount(p.getCount());
            } else {
                partialSum.add(p);
            }
        }

        if (partialSum != null) {
            context.write(key, partialSum);
        }
    }
}