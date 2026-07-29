package com.taxi.kmeans;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Point - a 10-dimensional vector representing one taxi trip (or one centroid).
 *
 * Implements Hadoop's Writable interface so that points can be serialized and
 * passed between the Mapper, Combiner and Reducer. A {@code count} field lets
 * the Combiner and Reducer aggregate partial sums of points: when several
 * points are summed together, {@code count} records how many original points
 * the sum represents, so the reducer can later divide by it to get the mean.
 */
public class Point implements Writable {

    /** Number of features per trip. */
    public static final int DIMENSIONS = 10;

    private double[] coords;
    private long count;

    /** No-arg constructor required by Hadoop's Writable deserialization. */
    public Point() {
        this.coords = new double[DIMENSIONS];
        this.count = 1;
    }

    /** Construct a point from a coordinate array (count defaults to 1). */
    public Point(double[] coords) {
        this.coords = coords;
        this.count = 1;
    }

    // ---------- accessors ----------

    public double[] getCoords() {
        return coords;
    }

    public long getCount() {
        return count;
    }

    public void setCount(long count) {
        this.count = count;
    }

    // ---------- parsing ----------

    /**
     * Parse a comma-separated line of {@value #DIMENSIONS} values into a Point.
     * The created point has a count of 1.
     */
    public static Point parse(String line) {
        String[] parts = line.split(",");
        double[] c = new double[DIMENSIONS];
        for (int i = 0; i < DIMENSIONS; i++) {
            c[i] = Double.parseDouble(parts[i].trim());
        }
        return new Point(c);
    }

    // ---------- math ----------

    /**
     * Squared Euclidean distance to another point. Squared distance is used
     * (no sqrt) because it is cheaper and preserves the ordering needed to
     * find the nearest centroid.
     */
    public double distanceTo(Point other) {
        double sum = 0.0;
        for (int i = 0; i < DIMENSIONS; i++) {
            double d = this.coords[i] - other.coords[i];
            sum += d * d;
        }
        return sum;
    }

    /**
     * Add another point into this one: sums each coordinate and adds the
     * counts. Used by the Combiner and Reducer to accumulate partial sums.
     */
    public void add(Point other) {
        for (int i = 0; i < DIMENSIONS; i++) {
            this.coords[i] += other.coords[i];
        }
        this.count += other.count;
    }

    /**
     * Return a new point equal to this summed point divided by its count,
     * i.e. the mean. The returned centroid has a count of 1.
     */
    public Point average() {
        double[] mean = new double[DIMENSIONS];
        for (int i = 0; i < DIMENSIONS; i++) {
            mean[i] = this.coords[i] / this.count;
        }
        return new Point(mean);
    }

    // ---------- Writable ----------

    @Override
    public void write(DataOutput out) throws IOException {
        for (int i = 0; i < DIMENSIONS; i++) {
            out.writeDouble(coords[i]);
        }
        out.writeLong(count);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        for (int i = 0; i < DIMENSIONS; i++) {
            coords[i] = in.readDouble();
        }
        count = in.readLong();
    }

    // ---------- output ----------

    /**
     * Render the point as a comma-separated line of its coordinates.
     * This is the format written for each new centroid and the format
     * {@link #parse(String)} reads back.
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < DIMENSIONS; i++) {
            if (i > 0) sb.append(',');
            sb.append(coords[i]);
        }
        return sb.toString();
    }
}
