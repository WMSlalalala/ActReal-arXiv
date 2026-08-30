package com.sensorworldmodel.collector;

import android.content.Context;
import android.net.Uri;

import org.json.JSONException;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public final class SessionExporter {
    private SessionExporter() {}

    public static void exportCurrent(Context context, Uri destination) throws IOException {
        exportCurrent(context, destination, StudyStore.sessionId(context));
    }

    public static void exportCurrent(
            Context context,
            Uri destination,
            String expectedSessionId) throws IOException {
        if (expectedSessionId == null || expectedSessionId.isEmpty()) {
            throw new IOException("No active session");
        }
        requireExpectedSession(context, expectedSessionId);
        if (StudyStore.isRecording(context)) {
            throw new IOException("Stop recording before export");
        }
        File source = StudyStore.sessionDir(context, expectedSessionId);
        if (!source.isDirectory()) {
            throw new IOException("Export session does not exist");
        }
        StudyStore.flushAndClosePendingWrites(context);
        requireExpectedSession(context, expectedSessionId);
        if (StudyStore.isRecording(context)) {
            throw new IOException("Recording started while preparing export");
        }
        try {
            StudyStore.writeExportAudit(context);
        } catch (JSONException error) {
            throw new IOException("Cannot write export audit", error);
        }
        requireExpectedSession(context, expectedSessionId);
        OutputStream raw = context.getContentResolver().openOutputStream(destination, "w");
        if (raw == null) {
            throw new IOException("Cannot open destination");
        }
        try (ZipOutputStream zip = new ZipOutputStream(
                new BufferedOutputStream(raw))) {
            addDirectory(zip, source, source.getName());
        }
    }

    private static void requireExpectedSession(Context context, String expectedSessionId)
            throws IOException {
        if (!expectedSessionId.equals(StudyStore.sessionId(context))) {
            throw new IOException("Active session changed during export");
        }
    }

    private static void addDirectory(ZipOutputStream zip, File file, String relative)
            throws IOException {
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children == null) {
                return;
            }
            for (File child : children) {
                addDirectory(zip, child, relative + "/" + child.getName());
            }
            return;
        }
        zip.putNextEntry(new ZipEntry(relative));
        try (BufferedInputStream input = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[64 * 1024];
            int count;
            while ((count = input.read(buffer)) >= 0) {
                zip.write(buffer, 0, count);
            }
        }
        zip.closeEntry();
    }
}
