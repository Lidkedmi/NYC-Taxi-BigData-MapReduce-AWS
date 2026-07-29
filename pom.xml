package com.taxi.kmeans;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * KMeans Reducer.
 * Receives all (partially summed) points for one centroid, computes their
 * mean, and emits the new centroid as a comma-separated Text line.
 * The output of this reducer becomes the centroid file for the next iteration.
 */
public class KMeansReducer extends Reducer<IntWritable, Point, IntWritable, Text> {

    @Override
    protected void reduce(IntWritable key, Iterable<Point> values, Context context)
            throws IOException, InterruptedException {

        Point sum = null;
        for (Point p : values) {
            if (sum == null) {
                sum = new Point(p.getCoords().clone());
                sum.setCount(p.getCount());
            } else {
                sum.add(p);
            }
        }

        if (sum != null) {
            Point newCentroid = sum.average();
            context.write(key, new Text(newCentroid.toString()));
        }
    }
}