package com.healthhub.exception;

/**
 * ResourceNotFoundException — thrown when a patient is not found.
 * Extends RuntimeException (unchecked) so we don't need try-catch everywhere.
 */
public class ResourceNotFoundException extends RuntimeException {
    public ResourceNotFoundException(String message) {
        super(message);
    }
}
