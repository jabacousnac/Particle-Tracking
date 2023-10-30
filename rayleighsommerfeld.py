# -*- coding: utf-8 -*-

import numpy as np
import h5py
import pyfftw

def rayleighsommerfeld(wavefront,
                       displacement,
                       wavelength=0.447,
                       magnification=0.048,
                       nozphase=False,
                       generate=False,
                       hanning=False):
    '''
    Numerically propagate wave with Rayleigh-Sommerfeld integral

    Convolves two-dimensional wavefront with Rayleigh-Sommerfeld
    propagator to estimate wavefront at one or more axial displacements.
    This is useful for numerically refocusing holograms.

    ...

    Arguments
    ---------
    wavefront: numpy.ndarray
        A two dimensional array of complex wavefront values.
        The mean (background) value is assumed to be 1 and
        the wave should be normalized accordingly.
    displacement: float | numpy.ndarray
        Displacement(s) from the focal plane [pixels].

    Keywords
    --------
    wavelength: float
        Wavelength of wave in medium [lengthscale units].
        Default: 0.447 um
    magnification: float
        Lengthscale units per pixel.
        Default: 0.135 um/pixel
    nozphase: bool
        Do not unwrap axial phase.
        Default: False
    generate: string
        Generate propagator? choose method
        Only necessary for running code first time.
    hanning: bool
        Apply two-dimensional Hanning window.
        Default: False

    Returns
    -------
    field: numpy.ndarray
        Complex wavefront at one or more planes specified by z
    '''

    if wavefront.ndim != 2:
        raise ValueError('wavefront must be two-dimensional')
    wavefront = np.array(wavefront, dtype=complex)
    ny, nx = wavefront.shape

    displacement = np.atleast_1d(displacement)
    result = np.zeros([ny, nx, len(displacement)], dtype=complex)

    # important factors
    k = 2.*np.pi * magnification/wavelength  # wavenumber [radians/pixel]

    # phase factor for Rayleigh-Sommerfeld propagator in Fourier space
    # Compute factor k*sqrt(1-qx**2+qy**2)
    # (FIXME MDH): Do I need to neglect the endpoint?
    qx = np.linspace(-0.5, 0.5, nx, endpoint=False, dtype=complex)
    qy = np.linspace(-0.5, 0.5, ny, endpoint=False, dtype=complex)
    qsq = qx ** 2 + (qy**2)[:, np.newaxis]
    qsq *= (wavelength / magnification) ** 2

    qfactor = k * np.sqrt(1. - qsq)

    if nozphase:
        qfactor -= k

    if hanning:
        qfactor *= np.sqrt(np.outer(np.hanning(ny), np.hanning(nx)))

    # Account for propagation and absorption
    ikappa = 1j * np.real(qfactor)
    gamma = np.imag(qfactor)

    # Go to Fourier space and apply RS propagation operator
    fft = pyfftw.interfaces.numpy_fft
    a = fft.ifft2(wavefront - 1.)           # offset for zero mean
    a = fft.fftshift(a)

    if generate == 'h5py_generate':
        save_propagator(wavefront, displacement, ikappa, gamma) #only needed the first time
        hf = h5py.File('Hqz.h5', 'r')
        h = hf.get('propagator')
        h = np.array(h)

    if generate == 'numpy_generate':
        h = np.load('Hqz.npy')

    for n, z in enumerate(displacement):
        #Hqz = h[:,:,n]
        Hqz = np.exp((ikappa * z - gamma * abs(z)))
        thisA = a * Hqz                        # convolve with propagator
        thisA = np.fft.ifftshift(thisA)        # shift center
        thisA = fft.fft2(thisA)             # transform back to real space
        result[:, :, n] = thisA                # save result

    return result + 1.                         # undo the previous offset.

def save_propagator(wavefront, displacement, ikappa, gamma):
    #save propagator in fourier space locally
    ny, nx = wavefront.shape
    nz = displacement.shape[0]
    Hqz = np.zeros([ny, nx, len(displacement)], dtype=complex)
    for n, z in enumerate(displacement):
        h = np.exp((ikappa * z - gamma * abs(z)))
        Hqz[:,:,n] = h
    #np.save('Hqz.npy', Hqz)
    hf = h5py.File('Hqz.h5', 'w')
    hf.create_dataset('propagator', data=Hqz)
    hf.close()
    return
