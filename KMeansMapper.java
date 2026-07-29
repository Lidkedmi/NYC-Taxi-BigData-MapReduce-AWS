package com.taxi.kmeans;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * KMeans Mapper.
 * For each input data point, finds the nearest centroid and emits
 * (centroidId, point). Centroids are loaded once in setup() from a file
 * whose path is passed via the job configuration.
 *
 * The centroid file may contain lines in either form:
 *   "c1,c2,...,c10"            (the initial centroids file)
 *   "id<TAB>c1,c2,...,c10"     (output of a previous reducer)
 * so an optional leading "id<TAB>" prefix is stripped before parsing.
 *
 * The centroid path may live on HDFS (hdfs://) or on cloud storage (s3://);
 * the FileSystem is resolved from the path itself so the same JAR runs
 * unchanged on a local Hadoop cluster and on AWS EMR.
 */
public class KMeansMapper extends Mapper<LongWritable, Text, IntWritable, Point> {

    private final List<Point> centroids = new ArrayList<>();
    private final IntWritable outKey = new IntWritable();

    @Override
    protected void setup(Context context) throws IOException {
        Configuration conf = context.getConfiguration();
        String centroidPath = conf.get("centroids.path");
        Path path = new Path(centroidPath);
        // Resolve the FileSystem for THIS path (HDFS or S3) instead of
        // assuming the cluster default.
        FileSystem fs = path.getFileSystem(conf);

        try (FSDataInputStream in = fs.open(path);
             BufferedReader reader = new BufferedReader(
                     new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                // Strip an optional "id<TAB>" prefix produced by a previous reducer.
                int tab = line.indexOf('\t');
                if (tab >= 0) {
                    line = line.substring(tab + 1);
                }
                centroids.add(Point.parse(line));
            }
        }
    }

    @Override
    protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {

        String line = value.toString().trim();
        if (line.isEmpty()) return;

        Point p;
        try {
            p = Point.parse(line);
        } catch (Exception e) {
            return;
        }

        int nearest = 0;
        double bestDist = Double.MAX_VALUE;
        for (int i = 0; i < centroids.size(); i++) {
            double dist = p.distanceTo(centroids.get(i));
            if (dist < bestDist) {
                bestDist = dist;
                nearest = i;
            }
        }

        outKey.set(nearest);
        context.write(outKey, p);
    }
}