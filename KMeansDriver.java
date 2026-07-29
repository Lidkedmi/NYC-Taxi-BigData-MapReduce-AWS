package com.taxi.kmeans;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * KMeans Driver - orchestrates the ITERATIVE MapReduce process on Hadoop.
 *
 * Each iteration is a full MapReduce job: it reads the current centroids,
 * assigns every point to its nearest centroid (Mapper), partially aggregates
 * them (Combiner), and recomputes centroids as the mean of assigned points
 * (Reducer). The driver loops until centroids stop moving (convergence) or a
 * maximum iteration count is reached.
 *
 * Paths may be HDFS paths (hdfs://) or cloud storage paths (s3://). Each path
 * resolves its own FileSystem via Path.getFileSystem(conf), so the same JAR
 * runs unchanged on a local Hadoop cluster and on AWS EMR.
 *
 * Usage: hadoop jar taxi-kmeans.jar
 *            <inputData> <initialCentroids> <outputDir> <maxIterations>
 */
public class KMeansDriver {

    private static final double CONVERGENCE_THRESHOLD = 1e-2;

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            System.err.println("Usage: KMeansDriver <input> <centroids> <output> <maxIter>");
            System.err.println("Received " + args.length + " arguments:");
            for (int i = 0; i < args.length; i++) {
                System.err.println("  arg[" + i + "] = " + args[i]);
            }
            System.exit(2);
        }

        String inputPath        = args[0];
        String initialCentroids = args[1];
        String outputDir        = args[2];
        int maxIterations       = Integer.parseInt(args[3]);

        Configuration conf = new Configuration();

        String currentCentroids = initialCentroids;
        int iteration = 0;
        boolean converged = false;

        while (iteration < maxIterations && !converged) {

            System.out.println("=== KMeans iteration " + iteration + " ===");

            Path iterOutput = new Path(outputDir + "/iteration_" + iteration);
            // Resolve the FileSystem for THIS path (HDFS or S3) instead of
            // assuming the cluster default. This is what makes the job
            // portable between a local Hadoop cluster and AWS EMR.
            FileSystem outFs = iterOutput.getFileSystem(conf);
            if (outFs.exists(iterOutput)) {
                outFs.delete(iterOutput, true);
            }

            conf.set("centroids.path", currentCentroids);

            Job job = Job.getInstance(conf, "KMeans iteration " + iteration);
            job.setJarByClass(KMeansDriver.class);

            job.setMapperClass(KMeansMapper.class);
            job.setCombinerClass(KMeansCombiner.class);
            job.setReducerClass(KMeansReducer.class);

            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Point.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(Text.class);

            FileInputFormat.addInputPath(job, new Path(inputPath));
            FileOutputFormat.setOutputPath(job, iterOutput);

            boolean success = job.waitForCompletion(true);
            if (!success) {
                System.err.println("Iteration " + iteration + " failed.");
                System.exit(1);
            }

            Path newCentroidFile = new Path(iterOutput, "part-r-00000");

            double movement = computeMovement(conf, currentCentroids, newCentroidFile);
            System.out.println("Iteration " + iteration
                    + " centroid movement = " + movement);

            if (movement < CONVERGENCE_THRESHOLD) {
                converged = true;
                System.out.println("Converged after iteration " + iteration);
            }

            currentCentroids = newCentroidFile.toString();
            iteration++;
        }

        System.out.println("=== KMeans finished after " + iteration
                + " iteration(s). Final centroids: " + currentCentroids + " ===");
    }

    /** Computes total squared movement between old and new centroids. */
    private static double computeMovement(Configuration conf, String oldPath, Path newPath)
            throws IOException {

        List<Point> oldC = readCentroids(conf, new Path(oldPath));
        List<Point> newC = readCentroids(conf, newPath);

        double total = 0.0;
        int n = Math.min(oldC.size(), newC.size());
        for (int i = 0; i < n; i++) {
            total += oldC.get(i).distanceTo(newC.get(i));
        }
        return total;
    }

    /**
     * Reads a centroid file, tolerating an optional leading id-TAB prefix.
     * The FileSystem is resolved from the path itself, so the file may live
     * on HDFS or on S3.
     */
    private static List<Point> readCentroids(Configuration conf, Path path)
            throws IOException {

        FileSystem fs = path.getFileSystem(conf);
        List<Point> result = new ArrayList<>();
        try (FSDataInputStream in = fs.open(path);
             BufferedReader reader = new BufferedReader(
                     new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                int tab = line.indexOf('\t');
                if (tab >= 0) {
                    line = line.substring(tab + 1);
                }
                result.add(Point.parse(line));
            }
        }
        return result;
    }
}