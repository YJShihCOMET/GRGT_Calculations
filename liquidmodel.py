import numpy as np
import pandas as pd
import molecule as mc
from scipy.constants import gas_constant, Avogadro, epsilon_0, elementary_charge
from abc import ABCMeta, abstractmethod
from COSMOSAC.easy_COSMOSAC import calc_LNAC

class Context():
    """
    The Context defines the interface of interest to clients.
    """
    
    def __init__(self, strategy):
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """
        self._strategy = strategy

    @property
    def strategy(self):
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy
        
    @property
    def mixture(self):
        """
        The Context maintains a reference to one of the mixture objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._mixture

    @mixture.setter
    def mixture(self, mixture):
        """
        Usually, the Context allows replacing a mixture object at runtime.
        """
        self._mixture = mixture
        self._strategy.set_mixture(mixture)
        
    def set_tx(self, t, x):
        """
        Parameters
        ----------
        t : float
            temperture (K)
        x : numpy array, 1Xn float
            mole fraction of component
            eg. x = np.array([x1, x2, ..., xn])
        """
        self.t = t
        self.x = x
        self._strategy.set_tx(t, x)
        
    def set_bip(self, a, b, c, d, e, f):
        """
        Parameters
        ----------
        a~f : numpy array, float nXn matrix 
            binary interaction parameters
        ----------
        Gij = exp(-alphaij * tauij)
        tauij = aij + bij / t + eij * ln(t) + fij * t
        alphaij = cij + dij * (t-273.15K)
        tauii = 0
        Gii = 1
        """  
        self._strategy.set_bip(a, b, c, d, e, f)  
        
    def set_epsilon(self, epsilon12dk, epsilon11dk, epsilon22dk, Nc=4, Nc_array=None):
        """
        Parameters
        ----------
        epsilon12dk, epsilon11dk, epsilon22dk: float
            potential energy for 1-2, 1-1, 2-2 pair (K)
        Nc : int
            coordination number
        """
        self._strategy.set_epsilon(epsilon12dk, epsilon11dk, epsilon22dk, Nc, Nc_array)  

    def set_cp(self, dbname, method):
        """ 
        Parameters
        ----------
        dbname : string
            name of database used in COSMOSAC model from MIT group (in class COSMOSAC_MIT)
            please set one of ['VT', 'UD', 'G09']
        method: string
            version of COSMOSAC model from MIT group (in class COSMOSAC_MIT)
            please set one of ['COSMOSAC-2002','COSMOSAC-2010','COSMOSAC-dsp']
            
        """ 
        self._strategy.set_cp(dbname, method)
        
    @property
    def nc(self):
        """ The number of component in the mixture """
        return self._strategy.nc
    
    @property
    def tauij(self):
        """ The NRTL and Wilson parameters for given mixture """
        return self._strategy._calc_tau()
    
    @property
    def Gij(self):
        """ The NRTL and Wilson parameters for given mixture """
        return self._strategy._calc_G()

    def u(self):
        """ The interaction potential for each given components """
        return self._strategy._get_u()
        
    def path(self):
        """ The path of COSMO file for each given components """
        return self._strategy.get_path()
    
    def tvc(self):
        """ Total volume of cavity for each components """
        return self._strategy._get_tvc()    
    
    def segnum(self):
        """ The Number of segments for each components """
        return self._strategy._get_segnum() 

    def cosmodata(self):
        """ The data of COSMO file for each components"""
        return self._strategy._get_cosmodata() 
    
    def sigmaarea(self):
        """ The sigma area of each components within 51 segments """
        return self._strategy._get_sigmaarea()
    
    def sumsigmaarea(self):
        """ The molecule area of each components """
        return self._strategy._get_sumsigmaarea()
    
    def sigmaprofile(self):
        """ The sigma profile of each components """
        return self._strategy._get_sigmaprofile()
    
    def dw(self):
        """ The exchange energy of each components """
        return self._strategy._get_dw()
    
    def seggamma(self):
        """ The segment activity coefficients of each pure components """
        return self._strategy._get_seggamma()
    
    def seggammam(self):
        """ The segment activity coefficients of given mixture """
        return self._strategy._get_seggammam()
    
    def sigmaprofilem(self):
        """ The sigma profile of given mixture """
        return self._strategy._get_sigmaprofilem()
    
    def gammaSG(self):
        """ The Staverman-Guggenheim activity coefficicent of given mixture """
        return self._strategy._get_gammaSG() 
        
    """
    The Context delegates some work to the Strategy object instead of
    implementing multiple versions of the algorithm on its own.
    """
    
    def calc_gammaminf(self):
        """ Start to calculate infintie dilution activity coefficients of each component within the given mixture """
        gammaminf = self._strategy._get_gammaminf()
        self.__gammaminf = gammaminf
        return gammaminf
    
    def calc_gammam(self):
        """ Start to calculate activity coefficients of each component within the given mixture """
        gammam = self._strategy._get_gammam()
        self.__gammam = gammam
        return gammam

    def calc_mu(self):
        """ Start to calculate chemical potentials of each component within the given mixture """
        mu = self._strategy._get_mu()
        self.__mu = mu
        return mu
    
    def calc_dmudxa(self):
        """ Start to calculate first derivatives of chemical potentials of each component (analytic) """
        dmudxa = self._strategy._get_dmudxa()
        self.__dmudxa = dmudxa
        return dmudxa
    
    def calc_dmudxn(self, dx=1.e-4):
        """ Start to calculate first derivatives of chemical potentials of each component (numerical) """
        dmudxn = self._strategy._get_dmudxn(dx)
        self.__dmudxn = dmudxn
        return dmudxn
        
    def calc_d2mudx2a(self):
        """ Start to calculate second derivatives of  chemical potentials of each component (analytic) """
        d2mudx2a = self._strategy._get_d2mudx2a()
        self.__d2mudx2a = d2mudx2a
        return d2mudx2a
    
    def calc_d2mudx2n(self, dx=1.e-4):
        """ Start to calculate first derivatives of chemical potentials of each component (numerical) """
        d2mudx2n = self._strategy._get_d2mudx2n(dx)
        self.__d2mudx2n = d2mudx2n
        return d2mudx2n
        
    def calc_gex(self):
        """ Start to calculate molar excess gibbs free energy of the given mixture """
        gex = self._strategy._get_gex()
        self.__gex = gex
        return gex
        
    def calc_hexa(self):
        """ Start to calculate molar excess enthalpy of the given mixture (analytic) """
        hexa = self._strategy._get_hexa()
        self.__hexa = hexa
        return hexa
    
    def calc_hexn(self, dt = 1.e-3):
        """ Start to calculate molar excess enthalpy of the given mixture (numerical) """
        hexn = self._strategy._get_hexn(dt)
        self.__hexn = hexn
        return hexn
    
    def calc_sexa(self):
        """ Start to calculate molar excess entropy of the given mixture (analytic) """
        sexa = self._strategy._get_sexa()
        self.__sexa = sexa
        return sexa
    
    def calc_sexn(self, dt = 1.e-3):
        """ Start to calculate molar excess entropy of the given mixture (numerical) """
        sexn = self._strategy._get_sexn(dt)
        self.__sexn = sexn
        return sexn
    
    def calc_dgmix(self):
        """ Start to calculate molar mixing gibbs free energy of the given mixture """
        dgmix = self._strategy._get_dgmix()
        self.__dgmix = dgmix
        return dgmix
    
    def calc_dsmixa(self):
        """ Start to calculate molar entropy of mixing of the given mixture (analytic) """
        dsmixa = self._strategy._get_dsmixa()
        self.__dsmixa = dsmixa
        return dsmixa
    
    def calc_dsmixn(self):
        """ Start to calculate molar entropy of mixing of the given mixture (numerical) """
        dsmixn = self._strategy._get_dsmixn()
        self.__dsmixn = dsmixn
        return dsmixn
    
    def calc_sa(self):
        """ Start to calculate molar entropy of the given mixture (analytic) """
        sa = self._strategy._get_sa()
        self.__sa = sa
        return sa
    
    def calc_sn(self):
        """ Start to calculate molar entropy of the given mixture (numerical) """
        sn = self._strategy._get_sn()
        self.__sn = sn
        return sn
        
    @property
    def get_gammam(self):
        """ Get the activity coefficient """
        return self.__gammam
    
    @property
    def get_gex(self):
        """ Get the molar excess gibbs free energy """
        return self.__gex
    
    @property
    def get_mu(self):
        """ Get the chemical potentials of each component """
        return self.__mu


def decorator_dmdtn(mi):
    """ 
    A decorator for numerically calculate first derivative of temperature of properties. 
    """
    def method(mth):
        def wrapper(self, dt):
            """
            Parameters
            ----------
            dti : float
                step size of temperature for the numerical differentiation
            ti : float
                temperture (K) 
            mi : function
                any property that is function of temperture
            mth : regular method
                the regular method you want to decorate it.
            mplusi : float or numpy array, float nX1 matrix
                property in the state of ti + dti
            mminusi : float or numpy array, float nX1 matrix
                property in the state of ti - dti
            dmdtni : float or numpy array, float nXn matrix
                first derivatives of the property (numerical equation)

            """
            dti = dt
            ti = self.t
            self.t = ti + dti
            mplusi = mi(self)
            self.t = ti - dti
            mminusi = mi(self)
            dmdtni = ( mplusi - mminusi ) / ( 2 * dti )
            self.t = ti
            return dmdtni
        return wrapper
    return method

def decorator_d2mdt2n(mi):
    """ 
    A decorator for numerically calculate second derivative of temperature of properties. 
    """
    def method(mth):
        def wrapper(self, dt):
            """
            Parameters
            ----------
            dti : float
                step size of temperature for the numerical differentiation
            ti : float
                temperture (K) 
            mi : function
                any property that is function of temperture
            mth : regular method
                the regular method you want to decorate it.
            mci : float or numpy array, float nX1 matrix
                property in the state of ti
            mplusi : float or numpy array, float nX1 matrix
                property in the state of ti + dti
            mminusi : float or numpy array, float nX1 matrix
                property in the state of ti - dti
            d2mdt2ni : float or numpy array, float nXn matrix
                second derivatives of the property (numerical equation)

            """
            dti = dt
            ti = self.t
            mci = mi(self)
            self.t = ti + dti
            mplusi = mi(self)
            self.t = ti - dti
            mminusi = mi(self)
            d2mdt2ni = ( mplusi - 2 * mci + mminusi ) / ( dti **2 )
            self.t = ti
            return d2mdt2ni
        return wrapper
    return method

def decorator_dmdx1n(mi):
    """ 
    A decorator for numerically calculate first derivative of x1 of properties. 
    """
    def method(mth):
        def wrapper(self, dx=1.e-4):
            """
            Parameters
            ----------
            dxi : float
                step size of mole fraction of component i for the numerical differentiation
            xi : numpy array, float nX1 matrix
                mole fraction of component i  
            mi : function
                any property that is function of mole fraction 
            mth : regular method
                the regular method you want to decorate it.
            mplusi : float or numpy array, float nX1 matrix
                property in the state of xi + dxi
            mminusi : float or numpy array, float nX1 matrix
                property in the state of xi - dxi
            dmdx1ni : float or numpy array, float nXn matrix
                first derivatives of the property (numerical equation)

            """
            dxi = dx
            xi = self.x
            x1i, x2i = xi
            self.x = np.array([x1i + dxi, x2i - dxi])
            mplusi = mi(self)
            self.x = np.array([x1i - dxi, x2i + dxi])
            mminusi = mi(self)
            dmdx1ni = ( mplusi - mminusi ) / ( 2 * dxi )
            self.x = xi
            return dmdx1ni
        return wrapper
    return method

def decorator_d2mdx12n(mi): 
    """ 
    A decorator for numerically calculate second derivative of x1 of properties. 
    """
    def method(mth):
        def wrapper(self, dx=1.e-4):
            """
            Parameters
            ----------
            dxi : float
                step size of mole fraction of component i for the numerical differentiation
            xi : numpy array, float nX1 matrix
                mole fraction of component i  
            mi : function
                any property that is function of mole fraction 
            mth : regular method
                the regular method you want to decorate it.
            mci : float or numpy array, float nX1 matrix
                property in the state of xi
            mplusi : float or numpy array, float nX1 matrix
                property in the state of xi + dxi
            mminusi : float or numpy array, float nX1 matrix
                property in the state of xi - dxi
            d2mdx12ni : float or numpy array, float nXn matrix
                second derivatives of the property (numerical equation)

            """
            dxi = dx
            xi = self.x
            mci = mi(self)
            x1i, x2i = xi
            self.x = np.array([x1i + dxi, x2i - dxi])
            mplusi = mi(self)
            self.x = np.array([x1i - dxi, x2i + dxi])
            mminusi = mi(self)
            d2mdx12ni = ( mplusi - 2 * mci + mminusi ) / ( dxi **2 )
            self.x = xi
            return d2mdx12ni
        return wrapper
    return method

class LiquidModelStrategy(metaclass=ABCMeta):
    
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """
    
    """
    Parameters
    ----------
    gpure : float
        molar Gibbs free energy of pure component i divide by RT (reference state)
    spure : float
        molar entropy of pure component i divide by R (reference state)
    """
    
    gpure = 0 
    spure = 0 
    dt = 1.e-3
    dx = 1.e-4
    
    def set_tx(self, t, x):
        """
        Parameters
        ----------
        t : float
            temperture (K)
        x : numpy array, float nX1 matrix
            mole fraction of component i  
            eg. x = np.array([x1, x2, ..., xn])
        """
        self.t = t
        self.x = x
    
    @abstractmethod
    def set_bip(self, a, b, c, d, e, f):
        pass
    
    @abstractmethod
    def set_mixture(self, mixture):
        pass
    
    @abstractmethod
    def _get_u(self):
        pass
    
    @abstractmethod
    def _get_gammam(self):
        pass
    
    def _get_mu(self):
        """
        Parameters
        ----------
        gpurei : float
            molar Gibbs free energy of pure component i
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        ti: float
            temperture (K)
        gammami : numpy array, float nX1 matrix
            activity coefficient of mixture of component i
        mui : numpy array, float nX1 matrix
            chemical potential of mixture of component i divide by RT

        """
        gpurei = self.gpure
        xi = self.x
        gammami = self._get_gammam()
        mui = gpurei + np.log(xi * gammami)
        return mui
    
    @abstractmethod
    def _get_dmudxa(self):
        pass
    
    @decorator_dmdx1n(mi=_get_mu)
    def _get_binary_dmudxn(self, dx=1.e-4):
        pass
    
    def _get_dmudxn(self, dx=1.e-4):
        """
        Parameters
        ----------
        nci : interger
            number of components
        dmu12dx1i : numpy array, float 1Xn matrix
            first derivatives of chemical potential of component 1 and 2 divide by RT (numerical equation)
            for binary systems: dmu12dx1i = np.array([dmu1dx1, dmu2dx1])
        dmudxi : numpy array, float nXn matrix
            first derivatives of chemical potential divide by RT (numerical equation)
            for binary systems: dmudxi = np.array([dmu1dx1, dmu2dx1], [dmu1dx2, dmu2dx2])
        """
        nci = self.nc
        if nci == 2:
            dmu12dx1i = self._get_binary_dmudxn(dx)
            dmudxi = np.array([[dmu12dx1i[0], dmu12dx1i[1]], [-dmu12dx1i[0], -dmu12dx1i[1]]])
            
        else:
            pass
        return dmudxi
    
    @abstractmethod
    def _get_d2mudx2a(self):
        pass
    
    @decorator_d2mdx12n(mi=_get_mu)
    def _get_binary_d2mudx2n(self, dx=1.e-4):
        pass
    
    def _get_d2mudx2n(self, dx=1.e-4):
        """
        Parameters
        ----------
        nci : interger
            number of components
        d2mu12dx12i : numpy array, float 1Xn matrix
            second derivatives of chemical potential of component 1 and 2 divide by RT (numerical equation)
            for binary systems: d2mu12dx12i = np.array([d2mu1dx12i, d2mu2dx12i])
        d2mudx2i : numpy array, float nXn matrix
            second derivatives of chemical potential divide by RT (analytic equation)
            for binary systems: dmudxi = np.array([d2mu1dx12, d2mu2dx12], [d2mu1dx22, d2mu2dx22])

        """
        nci = self.nc
        if nci == 2:
            d2mu12dx12i = self._get_binary_d2mudx2n(dx)
            d2mudx2i = np.array([[d2mu12dx12i[0], d2mu12dx12i[1]], [d2mu12dx12i[0], d2mu12dx12i[1]]])
            
        else:
            pass
        return d2mudx2i
    
    def _get_gex(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1
            mole fraction of component i  
        gammami : numpy array, float nX1 
            activity coefficient of mixture of component i
        gexi : float
            excess total molar Gibbs free energy divide by RT
        """
        xi = self.x
        if np.all(xi != 0):
            gammami = self._get_gammam()
            gexi = np.sum( xi * np.log(gammami) )
        else:
            gexi = 0
        return gexi
    
    @abstractmethod
    def _get_hexa(self):
        pass
    
    def _get_hexn(self, dt = 1.e-3):
        """
        Parameters
        ----------
        ti: float
            temperture (K)
        dti: float
            differential temperture (K)
        hexi : float
            excess total molar enthalpy divide by RT (numerical)

        """
        dti = dt
        ti = self.t
        self.t = ti + dti
        gexplusi = self._get_gex()
        self.t = ti - dti
        gexminusi = self._get_gex()
        hexi = - ti * ( gexplusi - gexminusi ) / ( 2 * dti )
        self.t = ti
        return hexi
    
    def _get_sexa(self):
        """
        Parameters
        ----------
        gexi : float
            excess total molar Gibbs free energy divide by RT
        hexi : float
            excess total molar enthalpy divide by RT (analytical)
        sexi : float
            excess total molar entropy divide by R (analytical)
        """
        gexi = self._get_gex()
        hexi = self._get_hexa()
        sexi = hexi - gexi
        return sexi
    
    def _get_sexn(self, dt = 1.e-3):
        """
        Parameters
        ----------
        dti: float
            differential temperture (K)
        gexi : float
            excess total molar Gibbs free energy divide by RT
        hexi : float
            excess total molar enthalpy divide by RT (numerical)
        sexi : float
            excess total molar entropy divide by R (numerical)
        """
        dti = dt
        gexi = self._get_gex()
        hexi = self._get_hexn(dti)
        sexi = hexi - gexi
        return sexi

    def _get_dgidmix(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1
            mole fraction of component i  
        dgidmixi : float
            total molar Gibbs free energy of ideal mixing divide by RT
        """
        xi = self.x
        dgidmixi = sum( xi * np.log( xi ) ) if np.all(xi != 0) else 0.
        return dgidmixi
    
    def _get_dgmix(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1
            mole fraction of component i  
        gexi : float
            excess total molar Gibbs free energy divide by RT
        dgidmixi : float
            total molar Gibbs free energy of ideal mixing divide by RT
        dgmixi : float
            total molar Gibbs free energy of mixing divide by RT
        """
        xi = self.x
        gexi = self._get_gex()
        dgidmixi = self._get_dgidmix()
        dgmixi = gexi + dgidmixi
        return dgmixi
    
    def _get_dsidmix(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1
            mole fraction of component i  
        dsidmixi : float
            total molar entropy of ideal mixing divide by RT
        """
        xi = self.x
        dsidmixi = - sum( xi * np.log( xi ) ) if np.all(xi != 0) else 0.
        return dsidmixi
    
    def _get_dsmixa(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        sexi : float
            excess total molar entropy divide by R (analytical)
        dsidmixi : float
            total molar entropy of ideal mixing divide by RT
        dsmixi : float
            total molar entropy of mixing divide by R (analytical)
        """
        xi = self.x
        sexi = self._get_sexa()
        dsidmixi = self._get_dsidmix()
        dsmixi = sexi + dsidmixi
        return dsmixi
    
    def _get_dsmixn(self, dt = 1.e-3):
        """
        Parameters
        ----------
        dti: float
            differential temperture (K)
        xi : numpy array, float nX1 matrix
            mole fraction of component i 
        spure : float
            molar entropy of pure component i divide by R
        sexi : float
            excess total molar entropy divide by R (numerical)
        dsidmixi : float
            total molar entropy of ideal mixing divide by RT
        dsmixi : float
            total molar entropy of mixing divide by R (numerical)
        """
        dti = dt
        xi = self.x
        sexi = self._get_sexn(dti)
        dsidmixi = self._get_dsidmix()
        dsmixi = sexi + dsidmixi
        return dsmixi
    
    def _get_sa(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        spure : float
            molar entropy of pure component i divide by R
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        dsmixi : float
            total molar entropy of mixing divide by R (analytical)
        si : float
            total molar entropy divide by R (analytical)
        """
        xi = self.x
        spurei = self.spure
        dsmixi = self._get_dsmixa()
        si = dsmixi + sum( xi * spurei )
        return si
    
    def _get_sn(self, dt = 1.e-3):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        spure : float
            molar entropy of pure component i divide by R
        dsmixi : float
            total molar entropy of mixing divide by R (numerical)
        si : float
            total molar entropy divide by R (numerical)
        """
        xi = self.x
        dti = dt
        spurei = self.spure
        dsmixi = self._get_dsmixn(dti)
        si = dsmixi + sum( xi * spurei )
        return si
    
    
"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class NRTL(LiquidModelStrategy):
          
    def set_bip(self, a, b, c, d, e, f):
        """
        Parameters
        ----------
        a~f : numpy array, float nXn matrix 
            binary interaction parameters
        ----------
        Gij = exp(-alphaij * tauij)
        tauij = aij + bij / t + eij * ln(t) + fij * t
        alphaij = cij + dij * (t-273.15K)
        tauii = 0
        Gii = 1
        """
        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d
        self.__e = e
        self.__f = f
        self.__check_bip()
        
    def set_mixture(self, mixture):
        self._mixture = mixture
        
    def __check_bip(self):
        """
        Parameters
        ----------
        aij~fij : numpy array, float nXn matrix
            binary interaction parameters
        nc : interger
            number of components
        """
        aij = self.__a
        bij = self.__b
        cij = self.__c
        dij = self.__d
        eij = self.__e
        fij = self.__f
        if np.shape(aij) == np.shape(bij) == np.shape(cij) == np.shape(dij) == np.shape(eij) == np.shape(fij) and \
           np.shape(aij)[0] == np.shape(aij)[1]:
            self.nc = np.shape(aij)[0]
        else:
            print('The shape of BIPs are not consistant.')
        
    def _calc_tau(self):
        """
        Parameters
        ----------
        aij~fij : numpy array, float nXn matrix
            binary interaction parameters
        ti: float
            temperture (K)
        tauij: numpy array, float nXn matrix
            one of binary interaction parameters

        """
        aij = self.__a
        bij = self.__b
        eij = self.__e
        fij = self.__f
        ti = self.t
        if np.all( eij == 0. ):
            tauij = aij + bij / ti + fij * ti
        else:
            tauij = aij + bij / ti + eij * np.log( ti ) + fij * ti
        return tauij
        
    def _calc_dtaudt(self):
        """
        Parameters
        ----------
        bij~fij : numpy array, float nXn matrix
            binary interaction parameters
        ti: float
            temperture (K)
        dtaudtij: numpy array, float nXn matrix
            one of temperature derivative of binary interaction parameters

        """
        bij = self.__b
        eij = self.__e
        fij = self.__f
        ti = self.t
        dtaudtij = - bij / ti **2 + eij / ti + fij
        return dtaudtij
    
    def _calc_alpha(self):
        """
        Parameters
        ----------
        cij, dij : numpy array, float nXn matrix
            binary interaction parameters
        ti: float
            temperture (K)
        alphaij: numpy array, float nXn matrix
            one of binary interaction parameters

        """
        cij = self.__c
        dij = self.__d
        ti = self.t
        alphaij = cij + dij * ( ti - 273.15 )
        return alphaij
        
    def _calc_G(self):
        """
        Parameters
        ----------
        tauij: numpy array, float nXn matrix
            one of binary interaction parameters
        alphaij: numpy array, float nXn matrix
            one of binary interaction parameters
        Gij: numpy array, float nXn matrix
            one of binary interaction parameters

        """
        tauij = self._calc_tau()
        alphaij = self._calc_alpha()
        Gij = np.exp( - alphaij * tauij )
        return Gij
        
    def _calc_dalphadt(self):
        """
        Parameters
        ----------
        dij : numpy array, float nXn matrix
            binary interaction parameters
        ti: float
            temperture (K)
        dalphadtij: numpy array, float nXn matrix
            one of temperature derivative of binary interaction parameters

        """
        dij = self.__d
        dalphadtij = dij
        return dalphadtij
        
    def _get_u(self):
        """
        Parameters
        ----------
        tauij: numpy array, float nXn matrix
            one of binary interaction parameters
        ui : float
            Interaction potential of the mixture divide by RT
        """
        tauij = self._calc_tau()
        ui = np.sum(tauij)
        return ui
        
    def _get_gammaminf(self):
        """
        Parameters
        ----------
        nci : interger
            number of components
        alphaij, tauij, Gij : numpy array, float nXn matrix
            one of binary interaction parameters.
        gammaminfi : numpy array, float nX1 matrix
            Infinite dilution activity coefficient of mixture of component i

        """
        nci = self.nc
        tauij = self._calc_tau()
        Gij = self._calc_G()
        if nci == 2:
            gammaminfi = np.exp(np.array([tauij[1,0] + tauij[0,1] * Gij[0,1], tauij[0,1] + tauij[1,0] * Gij[1,0]]))
        return gammaminfi
    
    def _get_gammam(self):
        """
        Parameters
        ----------
        n : interger
            number of component
        xi : numpy array, float nX1
            mole fraction of component i  
        Xi : numpy array, float nXn matrix
            mole fraction of component i
            eg. Xi=[[x1, x2, ...],[x1, x2, ...],...]
        tauij, Gij : numpy array, float nXn matrix
            one of binary interaction parameters.
        gammami : numpy array, float nX1 matrix
            activity coefficient of mixture of component i

        """
        n = len(self.x)
        xi = self.x.reshape(n, 1)
        Xi = np.array([self.x.T for i in range(n)])
        tauij = self._calc_tau()
        Gij = self._calc_G()
        gammami = np.exp( ( xi.T.dot( tauij * Gij ) / ( xi.T.dot( Gij ) ) ).T + \
                   np.array([np.diag( ( Gij * Xi / ( Xi.dot( Gij ) ) ).dot( ( tauij - \
                            ( Xi.dot( tauij * Gij ) )  / ( Xi.dot( Gij ) ) ).T ) ) ] ).T )
        return gammami[:,0]
    
    def _get_dmudxa(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i
        nci : interger
            number of components
        alphaij, tauij, Gij : numpy array, float nXn matrix
            one of binary interaction parameters.
        dmudxi : numpy array, float nXn matrix
            first derivatives of chemical potential divide by RT (analytic equation)
            for binary systems: dmudxi = np.array([dmu1dx1, dmu2dx1], [dmu1dx2, dmu2dx2])

        """
        xi = self.x
        nci = self.nc
        tauij = self._calc_tau()
        alphaij = self._calc_alpha()
        Gij = np.exp( - alphaij * tauij )
        dmudxi = np.zeros((nci,nci), dtype=float)
        if nci == 2:
            x1, x2 = xi[0], xi[1]
            tau12, tau21 = tauij[0,1], tauij[1,0]
            G12, G21 = Gij[0,1], Gij[1,0]
            dmu1dx1 = - 2 * x2 * ( tau21 * ( G21 / ( x1 + x2 * G21 ) ) **2 + G12 * tau12 / ( ( x2 + x1 * G12 ) **2) ) \
                          - 2 * x2 **2 * ( tau21 * G21 **2 * ( 1 - G21 ) / ( ( x1 + x2 * G21 ) **3 ) \
                                          + G12 * tau12 * ( G12 - 1 ) / ( ( x2 + x1 * G12 ) **3 ) ) + 1 / x1
            dmu2dx1 = np.nan
            dmu1dx2 = np.nan
            dmu2dx2 = np.nan
            dmudxi[0,0], dmudxi[0,1], dmudxi[1,0], dmudxi[1,1] = dmu1dx1, dmu2dx1, dmu1dx2, dmu2dx2
        else:
            pass
        return dmudxi
        
    def _get_d2mudx2a(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        nci : interger
            number of components
        alphaij, tauij, Gij : numpy array, float nXn matrix
            one of binary interaction parameters.
        d2mudx2i : numpy array, float nXn matrix
            second derivatives of chemical potential divide by RT (analytic equation)
            for binary systems: dmudxi = np.array([d2(mu1)d(x1)2, d2(mu2)d(x1)2], [d2(mu1)d(x2)2, d2(mu2)d(x2)2])

        """
        xi = self.x
        nci = self.nc
        tauij = self._calc_tau()
        alphaij = self._calc_alpha()
        Gij = np.exp( - alphaij * tauij )
        d2mudx2i = np.zeros((nci,nci), dtype=float)
        if nci == 2:
            x1, x2 = xi[0], xi[1]
            tau12, tau21 = tauij[0,1], tauij[1,0]
            G12, G21 = Gij[0,1], Gij[1,0]
            d2mu1dx12 = 2 * ( tau21 * ( G21 / ( x1 + x2 * G21 ) ) **2 + G12 * tau12 / ( ( x2 + x1 * G12) **2 ) ) + \
                        8 * x2 * ( tau21 * G21 **2 * ( 1 - G21 ) / ( ( x1 + x2 * G21 ) **3 ) + \
                                  G12 * tau12 * ( G12 - 1 ) / ( ( x2 + x1 * G12 ) **3) ) + \
                        6 * x2 **2 * ( tau21 * G21 **2 * ( 1 - G21 ) **2 / ( ( x1 + x2 * G21 ) **4 ) + \
                                      G12 * tau12 * ( G12 - 1 ) **2 / ( ( x2 + x1 * G12 ) **4 ) ) - 1 / ( x1 **2 )
            d2mu2dx12 = np.nan
            d2mu1dx22 = np.nan
            d2mu2dx22 = np.nan
            d2mudx2i[0,0], d2mudx2i[0,1], d2mudx2i[1,0], d2mudx2i[1,1] = d2mu1dx12, d2mu2dx12, d2mu1dx22, d2mu2dx22
        else:
            pass
        return d2mudx2i
    
    def _get_hexa(self):
        """
        Parameters
        ----------
        n : interger
            number of component
        xi : numpy array, float nX1
            mole fraction of component i  
        alphaij, tauij, Gij : numpy array, float nXn matrix
            binary interaction parameters.
        dalphadtij, dtaudtij, dGdtij: numpy array, float nXn matrix
            temperature derivative of binary interaction parameters
        hexi : float
            excess total molar enthalpy divide by RT

        """
        xi = self.x
        ti= self.t
        nci = self.nc
        tauij = self._calc_tau()
        alphaij = self._calc_alpha()
        Gij = np.exp( - alphaij * tauij )
        dtaudtij = self._calc_dtaudt()
        dalphadtij = self._calc_dalphadt()
        dGdtij = - np.exp( - alphaij * tauij ) * ( alphaij * dtaudtij + dalphadtij * tauij )
        if nci == 2:
            x1, x2 = xi[0], xi[1]
            tau12, tau21 = tauij[0,1], tauij[1,0]
            G12, G21 = Gij[0,1], Gij[1,0]
            dtaudt12, dtaudt21 = dtaudtij[0,1], dtaudtij[1,0]
            dGdt12, dGdt21 = dGdtij[0,1], dGdtij[1,0]
            hexi = - ti * x1 * x2 * ( ( x1 * ( dGdt21 * tau21 + G21 * dtaudt21 ) + x2 * G21 **2 * dtaudt21 ) / ( x1 + x2 * G21 ) **2 + \
                                     ( x2 * ( dGdt12 * tau12 + G12 * dtaudt12 ) + x1 * G12 **2 * dtaudt12 ) / ( x2 + x1 * G12 ) **2 )
        return hexi
    
class Wilson(LiquidModelStrategy):
          
    def set_bip(self, a, b, c, d, e, f):
        """
        Parameters
        ----------
        a~f : numpy array, float nXn matrix 
            binary interaction parameters
        ----------
        Gij = Omegaij = ( vi / vj ) * exp( - tauij )
        ln(Gij) = aij + bij / t + cij * ln(t) + dij * t + eij / t **2
        tauij  = ( Ncj / 2 ) * (epsilonij - epsilonjj) / kT  = aij + bij / t + cij * ln(t) + dij * t + eij / t **2
        alphaij = fij
        tauii = 0
        Gii = 1
        """
        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d
        self.__e = e
        self.__f = f
        self.__check_bip()
        
    def set_mixture(self, mixture):
        self._mixture = mixture
        
    def __check_bip(self):
        """
        Parameters
        ----------
        aij~fij : numpy array, float nXn matrix
            binary interaction parameters
        nc : interger
            number of components
        """
        aij = self.__a
        bij = self.__b
        cij = self.__c
        dij = self.__d
        eij = self.__e
        fij = self.__f
        if np.shape(aij) == np.shape(bij) == np.shape(cij) == np.shape(dij) == np.shape(eij) == np.shape(fij) and \
           np.shape(aij)[0] == np.shape(aij)[1]:
            self.nc = np.shape(aij)[0]
        else:
            print('The shape of BIPs are not consistant.')
        
    def _calc_tau(self):
        """
        Parameters
        ----------
        aij~eij : numpy array, float nXn matrix
            binary interaction parameters
        ti: float
            temperture (K)
        tauij: numpy array, float nXn matrix
            one of binary interaction parameters

        """
        aij = self.__a
        bij = self.__b
        cij = self.__c
        dij = self.__d
        eij = self.__e
        ti = self.t
        if np.all( cij == 0. ):
            tauij = - ( aij + bij / ti + dij * ti + eij / ti **2 )
        else:
            tauij = - ( aij + bij / ti + cij * np.log( ti ) + dij * ti + eij / ti **2 )
        return tauij
        
    def _calc_dtaudt(self):
        """
        Parameters
        ----------
        bij~eij : numpy array, float nXn matrix
            binary interaction parameters
        ti: float
            temperture (K)
        dtaudtij: numpy array, float nXn matrix
            one of temperature derivative of binary interaction parameters

        """
        bij = self.__b
        cij = self.__c
        dij = self.__d
        eij = self.__e
        ti= self.t
        dtaudtij = bij / ti **2 - cij / ti - dij + eij / ( 2 * ti **3 )
        return dtaudtij
    
    def _calc_alpha(self):
        """
        Parameters
        ----------
        fij : numpy array, float nXn matrix
            binary interaction parameters
        ti: float
            temperture (K)
        alphaij: numpy array, float nXn matrix
            one of binary interaction parameters

        """
        fij = self.__f
        ti= self.t
        alphaij = fij.copy()
        return alphaij
        
    def _calc_dalphadt(self):
        """
        Parameters
        ----------
        dalphadtij: numpy array, float nXn matrix
            one of temperature derivative of binary interaction parameters

        """
        nci = self.nc
        dalphadtij = np.zeros( (nc, nc), dtype=float )
        return dalphadtij
    
    def _calc_G(self):
        """
        Parameters
        ----------
        tauij: numpy array, float nXn matrix
            one of binary interaction parameters
        alphaij: numpy array, float nXn matrix
            one of binary interaction parameters
        Gij: numpy array, float nXn matrix
            one of binary interaction parameters
            Gij = Omegaij

        """
        tauij = self._calc_tau()
        alphaij = self._calc_alpha()
        Gij = np.exp( - alphaij * tauij )
        return Gij
        
    def _calc_dGdt(self):
        """
        Parameters
        ----------
        tauij, alphaij: numpy array, float nXn matrix
            one of binary interaction parameters
        dtaudtij, dalphadtij, dGdtij: numpy array, float nXn matrix
            one of temperature derivative of binary interaction parameters
            
        """
        tauij = self._calc_tau()
        alphaij = self._calc_alpha()
        dtaudtij = self._calc_dtaudt()
        dalphadtij = self._calc_dalphadt()
        dGdtij = - np.exp( - alphaij * tauij ) * ( alphaij * dtaudtij + dalphadtij * tauij )
        return dGdtij
        
    def _get_u(self):
        """
        Parameters
        ----------
        tauij: numpy array, float nXn matrix
            one of binary interaction parameters
        ui : float
            Interaction potential of the mixture divide by RT
            ui = sum(tauij) = sum_{i}( sum_{j}( Nc,j / 2 * ( epsilonij - epsilonjj ) / kT ) )
            
        """
        tauij = self._calc_tau()
        ui = np.sum(tauij)
        return ui
    
    def _get_gammam(self):
        """
        Parameters
        ----------
        nci : interger
            number of components
        xi : numpy array, float nX1
            mole fraction of component i  
        Xi : numpy array, float nXn matrix
            mole fraction of component i
            eg. Xi=[[x1, x2, ...],[x1, x2, ...],...]
        alphaij, Gij : numpy array, float nXn matrix
            one of binary interaction parameters.
        gammami : numpy array, float nX1 matrix
            activity coefficient of mixture of component i

        """
        nci = self.nc
        xi = self.x.reshape(1, nci)
        Xi = np.array([self.x for i in range(nci)])
        alphaij = self._calc_alpha()
        Gij = self._calc_G()
        GijT = Gij.T
        gammami = np.exp( ( 1 - np.log( xi.dot(GijT) ) - ( GijT / ( Xi.dot(GijT) ) ).dot(xi.T).T ) / alphaij[0,1] )
        return gammami[0,:]
    
    def _get_dmudxa(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        nci : interger
            number of components
        alphaij, Gij : numpy array, float nXn matrix
            one of binary interaction parameters.
        dmudxi : numpy array, float nXn matrix
            first derivatives of chemical potential divide by RT (analytic equation)
            for binary systems: dmudxi = np.array([dmu1dx1, dmu2dx1], [dmu1dx2, dmu2dx2])

        """
        xi = self.x
        nci = self.nc
        alphaij = self._calc_alpha()
        Gij = self._calc_G()
        dmudxi = np.zeros((nci,nci), dtype=float)
        if nci == 2:
            x1, x2 = xi[0], xi[1]
            G12, G21 = Gij[0,1], Gij[1,0]
            dmudxi[0,0] = ( ( G12 - 1 ) / ( x1 + x2 * G12 ) - G12 / ( x1 + x2 * G12 ) **2 + ( G21 / ( x1 * G21 + x2 ) )**2 )\
            / alphaij[0,1] + 1 / x1
            dmudxi[0,1] = np.nan
            dmudxi[1,0] = np.nan
            dmudxi[1,1] = np.nan
        else:
            pass
        return dmudxi
        
    def _get_d2mudx2a(self):
        """
        Parameters
        ----------
        xi : numpy array, float nX1 matrix
            mole fraction of component i  
        nci : interger
            number of components
        alphaij, Gij : numpy array, float nXn matrix
            one of binary interaction parameters.
        d2mudx2i : numpy array, float nXn matrix
            second derivatives of chemical potential divide by RT (analytic equation)
            for binary systems: dmudxi = np.array([d2(mu1)d(x1)2, d2(mu2)d(x1)2], [d2(mu1)d(x2)2, d2(mu2)d(x2)2])

        """
        xi = self.x
        nci = self.nc
        alphaij = self._calc_alpha()
        Gij = self._calc_G()
        d2mudx2i = np.zeros((nci,nci), dtype=float)
        if nci == 2:
            x1, x2 = xi[0], xi[1]
            G12, G21 = Gij[0,1], Gij[1,0]
            d2mu1dx12 = ( ( ( 1 - G12 ) / ( x1 + x2 * G12 ) ) **2 + 2 * G12 * ( 1 - G12 ) / ( x1 + x2 * G12 ) **3 -\
            2 * G21 **2 * ( G21 - 1 ) / ( x1 * G21 + x2 ) **3 ) / alphaij[0,1] - 1 / ( x1 **2 )
            d2mu2dx12 = np.nan
            d2mu1dx22 = np.nan
            d2mu2dx22 = np.nan
            d2mudx2i[0,0], d2mudx2i[0,1], d2mudx2i[1,0], d2mudx2i[1,1] = d2mu1dx12, d2mu2dx12, d2mu1dx22, d2mu2dx22
        else:
            pass
        return d2mudx2i
    
    def _get_hexa(self):
        """
        Parameters
        ----------
        n : interger
            number of component
        xi : numpy array, float nX1
            mole fraction of component i  
        alphaij, Gij : numpy array, float nXn matrix
            binary interaction parameters.
        dGdtij: numpy array, float nXn matrix
            temperature derivative of binary interaction parameters
        hexi : float
            excess total molar enthalpy divide by RT

        """
        xi = self.x
        ti= self.t
        nci = self.nc
        alphaij = self._calc_alpha()
        Gij = np.exp( - alphaij * tauij )
        dGdtij = self._calc_dGdt()
        if nci == 2:
            x1, x2 = xi[0], xi[1]
            G12, G21 = Gij[0,1], Gij[1,0]
            dGdt12, dGdt21 = dGdtij[0,1], dGdtij[1,0]
            hexi = ti * x1 * x2 * ( dGdt12 / ( x1 + x2 * G12 ) + dGdt21 / ( x1 * G21 + x2 ) ) / alphaij[0,1]
        return hexi
    
    
            
class COSMOSACStrategy(LiquidModelStrategy):
    """
    Parameters
    ----------
    path_dict : dictionary, key:string ; value:string
        The paths of COSMO file for each components
    """ 
    path_dict = {
        'Water':'Data\\COSMO file\\WATER-00174.DMol3.cosmo', 
        'Acetone':'Data\\COSMO file\\ACETONE-00004.DMol3.cosmo', 
        'Methane':'Data\\COSMO file\\METHANE-01051.DMol3.cosmo',
        'n-Hexane':'Data\\COSMO file\\n-HEXANE-00089.DMol3.cosmo',
        'Cyclopentane':'Data\\COSMO file\\CYCLOPENTANE-00051.DMol3.cosmo',
        'Cyclohexane':'Data\\COSMO file\\CYCLOHEXANE-00050.DMol3.cosmo',
        'Benzene':'Data\\COSMO file\\BENZENE-00031.DMol3.cosmo',
        'Cyclohexanone':'Data\\COSMO file\\CYCLOHEXANONE-00250.DMol3.cosmo',
        'Methanol':'Data\\COSMO file\\METHANOL-00110.DMol3.cosmo',
        '1-Butanol':'Data\\COSMO file\\1-BUTANOL-00039.DMol3.cosmo',
        '1-Octanol':'Data\\COSMO file\\1-OCTANOL-00344.DMol3.cosmo',
        '1-Nonanol':'Data\\COSMO file\\1-NONANOL-00127.DMol3.cosmo',
        'p-Chlorophenol':'Data\\COSMO file\\p-CHLOROPHENOL-00812.DMol3.cosmo',
        'Monoethanolamine':'Data\\COSMO file\\MONOETHANOLAMINE-00546.DMol3.cosmo',
        'Nitromethane':'Data\\COSMO file\\NITROMETHANE-00125.DMol3.cosmo',
        'N-Formylmorpholine':'Data\\COSMO file\\N-FORMYLMORPHOLINE-01223.DMol3.cosmo',
        'Aceticanhydride':'Data\\COSMO file\\ACETICANHYDRIDE-00233.DMol3.cosmo'
    }
    
    dirpath = 'Data\\COSMO file'
    
    def set_bip(self, a, b, c, d, e, f):
        print("Error: You inputed the BIPs into COSMOSAC model.")

    def set_mixture(self, mixture):
        try:
            flag_mixture = ( self._mixture == mixture ) 
        except AttributeError as error:
            flag_mixture = False
        finally:
            self._mixture = mixture
            self.__check_mixturename()
            self.__sigmaarea = self.__sigmaarea if flag_mixture is True else None
            self.__dw = self.__dw if flag_mixture is True else None
            
    def _get_u(self):
        """
        Parameters
        ----------
        nci : interger
            number of components
        hexi : float
            excess total molar enthalpy divide by RT (numerical)
        zci : integer
            The Coordinate Number around each component (zci = 2 in NRTL model and Wilson model)
        ui : float
            Interaction potential of the mixture divide by RT
        """
        nci = self.nc
        hexi = self._get_hexn()
        zci = 2
        if nci == 2:
            ui = hexi * 8 / zci
        else:
            ui = np.nan
        return ui
    
    def __check_mixturename(self):
        """
        Parameters
        ----------
        nc : interger
            The Number of components
        mixture : list, string (1Xn)
            Component names in the mixture
        """ 
        mixture = self._mixture
        for component in mixture:
            if type(component) != str:
                print(f'The type of component {component} is not str.')
        else:
            self.nc = len(mixture)
    
    def _get_dmudxa(self):
        print('COSMOSAC model can not analytically calculate first derivatives of chemical potential.')
        
    def _get_d2mudx2a(self):
        print('COSMOSAC model can not analytically calculate second derivatives of chemical potential.')
    
    def _get_hexa(self):
        print('COSMOSAC model can not analytically calculate excess total molar enthalpy.')
    
    
class COSMOSAC2002e(COSMOSACStrategy):
    
    """
    Parameters to convert units
    ----------
    bohr_angstrom : float
        Convert Bohr(au) to Angstrom(A)
    kcal_jole : float
        Convert kilocalorie(kcal) to Jole(J)
      
    """ 
    __bohr_angstrom = 0.529177249
    __kcal_jole = 4184.
    
    """
    Parameters to calculate Sigma Profile
    ----------
    segaeff : float
        The standard segment surface area (A**2)
    fdecay : float 
        The unit conversion parameter, to correct the distance dmn from A to Bohr radius
    sigma_nnum : integer 
        The number of segment after splitting
    sigma_upper : float
        The maximum value of charge density after splitting (e/A**2)
    sigma_lower : float
        The minimum value of charge density after splitting (e/A**2)
    sigma_narray : numpy array, float (1Xsigma_nnum)
        The charge densities that are original segment areas split into (e/A**2)
    """ 
    __segaeff = 7.5
    __fdecay = 3.571064 # paper:0.52918**(-2)
    __sigma_nnum = 51
    __sigma_upper = 0.025
    __sigma_lower = -0.025
    __sigma_narray = np.linspace(__sigma_lower, __sigma_upper, __sigma_nnum)
    
    """ 
    Parameters to calculate exchange energy
    ----------
    na : float
        Avogadro constant (1/mol)
    electron_volt : float
        electron volt (J)
    epsilon0cnm : float
        The permittivity of free space (C**2/N**2-m)
    epsilon0 : float
        The permittivity of free space (e**2*mol/kcal-A)
    epsilon : float
        The parameter to calculate fpol
    ke : float
        The Coulomb constant, ( = 1/(4*pi*epsilon) ) (kcal*A/mol-e**2)
    fpol : float
        The parameter to calculate c_hb and alphap
    sigma_hb : float
        a cutoff value for hydrogen-bonding interactions (e/A**2)
    a_hb : float
        The parameter to calculate c_hb (A**2)
    c_hb : float
        a constant for the hydrogen-bonding interaction (kcal*A**4/e**2-mol)
    alphap : float
        a constant for the misfit energy (kcal*A**4/e**2-mol)
    
    """ 
    __na = Avogadro # 6.02214076e+23
    __electron_volt = elementary_charge # 1.602176634e-19
    __epsilon0cnm = epsilon_0 # 8.854187817620389e-12
    __epsilon0 = __epsilon0cnm * __kcal_jole / ( __electron_volt **2 * 1.e10 * __na ) # 2.3964518977e-4 , paper:2.395e-4
    __ke = 1 / ( 4 * np.pi * __epsilon0 )
    __auu = 1.07
    __fpol = 0.6916 # paper:( epsilon - 1 ) / ( epsilon + 0.5 ), epsilon = 3.667
    __sigma_hb = 0.0084
    __a_hb = 35.722
    __c_hb = np.sqrt( 4 * np.pi ) * __fpol * __auu * __ke * __a_hb **( 3 / 2 ) / 2 # 92990.53957275, paper: 85580.
    __alphap = np.sqrt( 4 * np.pi ) * __fpol * __auu * __ke * __segaeff **( 3 / 2 ) # 17891.92253736, paper: 16466.72
    
    """ 
    Parameters to calculate segment activity coefficient
    ----------
    k : float
        The constant of gas (kcal/mol-K)
    tol_seggamma : float
        The tolerance of calculation of segment activity coefficient
    
    """ 
    k = gas_constant / __kcal_jole
    tol_seggamma = 1.e-12
    
    """ 
    Parameters to calculate Staverman-Guggenheim activity coefficient
    ----------
    coornum : integer
        The Coordinate Number
    normarea : float
        The normalized standard surface area parameter (A**2)
    normvolume : float
        The normalized standard volume parameter (A**3)
    
    """ 
    __coornum = 10
    __normarea = 79.531954 # paper: 79.53
    __normvolume = 66.693927 # paper: 66.69
    
    """ 
    Parameters to determine whether some properties have calculated
    ----------
    sigmaarea : None or not None
        Whether "the areas of each segments for each components after splitting" have been calculated
    dw : None or not None
        Whether the exchange energy have been calculated
    
    """ 
    __sigmaarea = None
    __dw = None
    

    
    def _get_mixobj(self):
        """
        Parameters
        ----------
        dirpath : string
            Directory path of COSMO file
        mixture : list, string (1Xn)
            Component names in the mixture
        nci : list, string (1Xn)
            The path of COSMO file for each components
        mixobj : object
            The object of class "COSMOData_Dmol3" in module "molecule"
        """ 
        dirpath = self.dirpath
        mixture = self._mixture
        nci = self.nc
        # path = [self.path_dict.get(component,'Unknown') for component in mixture]
        mixobj = [None for itype in range(nci)]
        for i in range(nci):
            mixobj[i] = mc.COSMOData_DMol3(compname=mixture[i], dirpath=dirpath)
        return mixobj
    
    '''
    @staticmethod
    def __get_linenum(strval, path):
        """
        Parameters
        ----------
        strval : string
            The keyword
        path : string
            The path of file
        linenum : interger
            The number of lines of the keyword in the file
        """ 
        with open(path, 'r') as file:
            for linenum, line in enumerate(file, start=1):
                if strval in line.strip():
                    break
        return linenum

    @staticmethod
    def __get_linecontent(strval, path):
        """
        Parameters
        ----------
        strval : string
            The keyword
        path : string
            The path of file
        line : string
            The content of lines of the keyword in the file
        """ 
        with open(path, 'r') as file:
            for linenum, line in enumerate(file, start=1):
                if strval in line.strip():
                    break
        return line
    '''
    
    def _get_tvc(self):
        """
        Parameters
        ----------
        mixobj : object
            The object of class "COSMOData_Dmol3" in module "molecule"
        tvc : numpy array, float (1Xn)
            The total volume of cavity for each components (A**3)
        """ 
        mixobj = self._get_mixobj()
        tvc = np.array([compobj.tvc for compobj in mixobj]) 
        return tvc
     
    def _get_segnum(self):
        """
        Parameters
        ----------
        mixobj : object
            The object of class "COSMOData_Dmol3" in module "molecule"
        segnum : numpy array, integer (1Xn)
            The number of segments for each components 
        """ 
        mixobj = self._get_mixobj()
        segnum = np.array([compobj.segnum for compobj in mixobj]) 
        return segnum
    
    def _get_cosmodata(self):
        """
        Parameters
        ----------
        mixobj : object
            The object of class "COSMOData_Dmol3" in module "molecule"
        segnum : numpy array, integer (1Xn)
            The number of segments for each components 
        df_cosmo : n pandas dataframes, float (nXsegnumX9)
            The data frames of COSMO data for each components
        """ 
        mixobj = self._get_mixobj()
        cosmodata = [compobj.cosmodata for compobj in mixobj]
        return cosmodata
    
    @staticmethod
    def __lever_fun(x_middle, y_middle, x_lower, x_upper):
        """
        Parameters
        ----------
        x_lower : float
            The lower value of x
        x_middle : float
            The input value of x
        x_upper : float
            The upper value of x
        y_lower : float
            The lower value of y
        y_middle : float
            The input value of y
        y_upper : float
            The upper value of y
        ----------
        Implement lever rule by solve following 2 eqautions:
        x_middle * y_middle = x_lower * y_lower + x_upper * y_upper
        y_middle = y_lower + y_upper
        In the calculation of sigmaarea, x = charg density, y = segment area
        """ 
        y_upper = ( x_middle - x_lower ) / ( x_upper - x_lower ) * y_middle
        y_lower = y_middle - y_upper
        return y_upper, y_lower
        
    def _get_sigmaarea(self):
        """
        Parameters
        ----------
        cosmodata : pandas dataframe, float (nXsegnumX9)
            The data frames of COSMO data for each components
        segnum : numpy array, integer (1Xn)
            The number of segments for each components 
        segreff2 : float
            The square of radius of the standard segment surface area (A**2)
        sigma_nnum : integer
            The number of segment after splitting
        fdecay : float
            The unit conversion parameter, to correct the distance dmn from A to Bohr radius
        sigma_narray : numpy array, float (1Xsigma_nnum)
            The charge densities that are original segment areas split into (e/A**2)
        cosmodata_sigma : numpy array, float (1Xsegnum)
            The orginal charge densities of each segments in COSMO files (e/A**2)
        cosmodata_segposition : numpy array, float (segnumX3)
            The central positions of each segments in COSMO files (A)
        cosmodata_segarea : numpy array, float (1Xsegnum)
            The original segment areas of each segments in COSMO files (A**2)
        cosmodata_segrn2 : numpy array, float (1Xsegnum)
            The square of radius of original segment of each segments in COSMO files (A**2)
        cosmodata_sigmam : numpy array, float (1Xsegnum)
            The charge densities of each segments after averaging (e/A**2)
        sigmaareai : numpy array, float (1Xsigma_nnum)
            The areas of each segments after splitting (A**2)
        sigmaarea : numpy array, float (nXsigma_nnum)
            The areas of each segments for each components after splitting (A**2)
        """ 
        if self.__sigmaarea is None: # if sigmaarea have not been calculated, calculate it
            cosmodata = self._get_cosmodata()
            segnum = self._get_segnum()
            segreff2 = self.__segaeff / np.pi
            sigma_nnum = self.__sigma_nnum
            fdecay = self.__fdecay
            sigma_narray = self.__sigma_narray
            sigmaarea = []
#             print(cosmodata, segnum, segreff2, sigma_nnum, fdecay, sigma_narray)
            for cosmodatai, segnumi in zip(cosmodata, segnum):
                # cosmodata_sigma =  np.array(cosmodatai['charge(e)'] / cosmodatai['area(A**2)']) # one dimension (vary segment)
                cosmodata_sigma =  np.array(cosmodatai['charge/area(e/A**2)']) # one dimension (vary segment)
                cosmodata_segposition = np.array(cosmodatai.iloc[:,[2,3,4]]) # two dimension (vary segment, (x,y,z))
                cosmodata_segarea = np.array(cosmodatai['area(A**2)']) # one dimension (vary segment)
                cosmodata_segrn2 = cosmodata_segarea / np.pi
                cosmodata_sigmam = np.zeros(segnumi, dtype=float)  # mean sigma
                sigmaareai = np.zeros(sigma_nnum, dtype=float)

                # Averaging
                for n in range(segnumi):
                    segdmn2 = np.sum((cosmodata_segposition[n,:] - cosmodata_segposition) **2, axis=1)
                    avefactor = cosmodata_segrn2 * segreff2 / ( cosmodata_segrn2 + segreff2 ) * \
                                np.exp( - fdecay * segdmn2 / ( cosmodata_segrn2 + segreff2 ) )
                    cosmodata_sigmam[n] = np.sum(cosmodata_sigma * avefactor) / np.sum(avefactor)

                # Using lever rule to split area of segment into two sides of seg_sigma
                for n in range(segnumi): # Run segment in COSMO file
                    for n50 in range(sigma_nnum-1): # Run sigma=-0.025~0.024
                        if ( sigma_narray[n50+1] > cosmodata_sigmam[n] >= sigma_narray[n50] ):
                            lever_sol = self.__lever_fun( cosmodata_sigmam[n], cosmodata_segarea[n]\
                                                         , sigma_narray[n50], sigma_narray[n50+1] )
                            sigmaareai[n50] += lever_sol[1]
                            sigmaareai[n50+1] += lever_sol[0]
                            break
                        elif ( cosmodata_sigmam[n] == sigma_narray[sigma_nnum-1] ): #sigma=0.025
                            sigmaareai[sigma_nnum-1] += cosmodata_segarea[n]
                            break
                sigmaarea.append(sigmaareai)
            self.__sigmaarea = np.array(sigmaarea)
            return self.__sigmaarea
        else: # if sigmaarea have been calculated, return it
            return self.__sigmaarea
    
    def _get_sumsigmaarea(self):
        """
        Parameters
        ----------
        sigmaarea : numpy array, float (nXsigma_nnum)
            The areas of each segments for each components after splitting (A**2)
        sumsigmaarea : numpy array, float (1Xn)
            The summation of areas of each segments (ie. the areas of the molecule) for each components (A**2)
        """ 
        sigmaarea = self._get_sigmaarea()
        sumsigmaarea = np.array([np.sum(sigmaareai) for sigmaareai in sigmaarea])
        return sumsigmaarea
    
    def _get_sigmaprofile(self):
        """
        Parameters
        ----------
        sigmaarea : numpy array, float (nXsigma_nnum)
            The areas of each segments for each components after splitting (A**2)
        sumsigmaarea : numpy array, float (1Xn)
            The summation of areas of each segments (ie. the areas of the molecule) for each components (A**2)
        sigmaprofile : numpy array, float (nXsigma_nnum)
            The probability of finding a segment with each surface charge density in pure liquid for each components 
        """ 
        sigmaarea = self._get_sigmaarea()
        sumsigmaarea = self._get_sumsigmaarea()
        sigmaprofile = np.array([sigmaareai / sumsigmaareai for sigmaareai, sumsigmaareai in zip(sigmaarea, sumsigmaarea)])
        return sigmaprofile

    def _get_sigmaprofilem(self):
        """
        Parameters
        ----------
        xi : numpy array, float (1Xn) 
            The mole fraction of each components
        sumsigmaarea : numpy array, float (1Xn)
            The summation of areas of each segments (ie. the areas of the molecule) for each components (A**2)
        sigmaprofile : numpy array, float (nXsigma_nnum)
            The probability of finding a segment with each surface charge density in pure liquid for each components  
        sigmaprofilem : numpy array, float (1Xsigma_nnum)
            The probability of finding a segment with each surface charge density in the mixture
        """ 
        xi = self.x
        sumsigmaarea = self._get_sumsigmaarea()
        sigmaprofile = self._get_sigmaprofile()
        sigmaprofilem = np.array([np.sum( xi * sumsigmaarea * sigmaprofilei ) / np.sum( xi * sumsigmaarea ) \
                         for sigmaprofilei in zip(*sigmaprofile)]) 
        # map(list, zip(*sigmaprofile)):Transpose, then transform into generator
        return sigmaprofilem
    
    def _get_dw(self):
        """ 
        Parameters
        ----------
        sigma_hb : float
            a cutoff value for hydrogen-bonding interactions (e/A**2)
        c_hb : float
            a constant for the hydrogen-bonding interaction (kcal*A**4/e**2-mol)
        alphap : float
            a constant for the misfit energy (kcal*A**4/e**2-mol)
        sigma_narray : numpy array, float (1Xsigma_nnum)
            The charge densities that are original segment areas split into (e/A**2)
        dw : numpy array, float (sigma_nnumXsigma_nnum)
            The exchange energy (kcal/mol)
        """ 
        if self.__dw is None: # if dw have not been calculated, calculate it
            sigma_hb = self.__sigma_hb
            c_hb = self.__c_hb
            alphap = self.__alphap  # paper: 0.3 * fpol * segaeff **1.5 / epsilon0  # unit: kcal*au**4 / (e**2*mol)
            sigma_narray = self.__sigma_narray
            dw = np.zeros( (self.__sigma_nnum, self.__sigma_nnum), dtype=float )
            for m, sigma_m in enumerate(sigma_narray):
                for n, sigma_n in enumerate(sigma_narray):
                    (sigma_acc, sigma_don) = (sigma_m, sigma_n) if sigma_m >= sigma_n else (sigma_n, sigma_m)
                    dw[m,n] = ( alphap / 2 ) * ( sigma_n + sigma_m ) **2 + \
                                c_hb * max([0, (sigma_acc - sigma_hb)]) * min([0, (sigma_don + sigma_hb)])
            self.__dw = dw
            return dw
        else:# if dw have been calculated, return it
            return self.__dw
    '''
    def _seggamma_fun(self, sigmaprofile):
        """ 
        Parameters
        ----------
        ti: float
            temperture (K)
        k : float
            The constant of gas (kcal/mol-K)
        tol_seggamma : float
            The tolerance of calculation of segment activity coefficient
        sigma_nnum : integer
            The number of segment after splitting
        dw : numpy array, float (sigma_nnumXsigma_nnum)
            The exchange energy (kcal/mol)
        gamma_n : numpy array, float (for mixture:1Xsigma_nnum, for pure:nXsigma_nnum)
            The segment activity coefficients of each segment
        """ 
        ti = self.t
        k = self.k
        tol_seggamma = self.tol_seggamma
        sigma_nnum = self.__sigma_nnum
        dw = self._get_dw()
        gamma_n = np.ones( sigma_nnum, dtype=float )
        while True:
            gamma_m = np.array([1 / np.sum( sigmaprofile * gamma_n * np.exp( - dwm / ( k * ti ) ) ) for dwm in dw])
            eg = sum((np.log(gamma_m) - np.log(gamma_n))**2)
            if eg >= tol_seggamma: 
                gamma_n *= ( gamma_m / gamma_n ) **0.75 
            else:
                break
        return gamma_n
    '''
    def _seggamma_fun(self, sigmaprofile):
        t = self.t
        k = self.k
        tol_seggamma = self.tol_seggamma
        sigma_nnum = self.__sigma_nnum
        dw = self._get_dw()
        gamma_n = np.ones( sigma_nnum, dtype=float )
        while True:
            gamma_m = np.array([1 / np.sum( sigmaprofile * gamma_n * np.exp( - dwm / ( k * t ) ) ) if sigmaprofilem != 0. else 1. \
                                for dwm, sigmaprofilem in zip(dw,sigmaprofile)])
            eg = sum((np.log(gamma_m) - np.log(gamma_n))**2)
            # print(f'T={t} K, error={eg}')
            if eg >= tol_seggamma: 
                gamma_n *= ( gamma_m / gamma_n ) **0.75 
            else:
                break
        return gamma_n
    
    def _get_seggamma(self):
        """
        Parameters
        ----------
        sigmaprofile : numpy array, float (nXsigma_nnum)
            The probability of finding a segment with each surface charge density in pure liquid for each components  
        seggamma : numpy array, float (nXsigma_nnum)
            The segment activity coefficients of each segment for each pure component
        """ 
        sigmaprofile = self._get_sigmaprofile()
        seggamma = np.array([self._seggamma_fun(sigmaprofilei) for sigmaprofilei in sigmaprofile])
        return seggamma
            
    def _get_seggammam(self):
        """
        Parameters
        ----------
        sigmaprofilem : numpy array, float (1Xsigma_nnum)
            The probability of finding a segment with each surface charge density in pure liquid for each components  
        seggammam : numpy array, float (1Xsigma_nnum)
            The segment activity coefficients of each segment for the mixture
        """ 
        sigmaprofilem = self._get_sigmaprofilem()
        seggammam = self._seggamma_fun(sigmaprofilem)
        return seggammam
    
    def _get_gammaSG(self):
        """ 
        Parameters
        ----------
        xi : numpy array, float (1Xn) 
            The mole fraction of each components
        coornum : integer
            The Coordinate Number
        normarea : float
            The normalized standard surface area parameter(A**2)
        normvolume : float
            The normalized standard volume parameter (A**3)
        tvc : numpy array, float (1Xn)
            The total volume of cavity for each components (A**3)
        sumsigmaarea : numpy array, float (1Xn)
            The summation of areas of each segments (ie. the areas of the molecule) for each components (A**2)
        normr : numpy array, float (1Xn)
            The ratio of tvc and normvolume
        normq : numpy array, float (1Xn)
            The ratio of sumsigmaarea and normarea
        theta : numpy array, float (1Xn)
            The molar weighted segment for each components
        phi : numpy array, float (1Xn)
            The molar area fractional components for each components
        l : numpy array, float (1Xn)
            The compound parameter of normr for each components
        gammaSG : numpy array, float (1Xn)
            The Staverman-Guggenheim activity coefficient for each components
        """ 
        xi = self.x
        coornum = self.__coornum
        normarea = self.__normarea
        normvolume = self.__normvolume
        tvc = self._get_tvc()
        sumsigmaarea = self._get_sumsigmaarea()
        normr = tvc / normvolume
        normq = sumsigmaarea / normarea
        theta = xi * normq / np.sum( xi * normq )
        phi = xi * normr / np.sum( xi * normr )
        l = coornum / 2 * ( normr - normq ) - ( normr - 1 )
        gammaSG = np.exp( np.log( phi / xi ) + ( coornum / 2 * normq ) * np.log( theta / phi ) + l - phi / xi * np.sum( xi * l ) )
        return gammaSG

    def _get_gammam(self):
        """ 
        Parameters
        ----------
        segaeff : float
            The standard segment surface area (A**2)
        sigmaprofile : numpy array, float (nXsigma_nnum)
            The probability of finding a segment with each surface charge density in pure liquid for each components  
        sigmaprofilem : numpy array, float (1Xsigma_nnum)
            The probability of finding a segment with each surface charge density in pure liquid for each components  
        sumsigmaarea : numpy array, float (1Xn)
            The summation of areas of each segments (ie. the areas of the molecule) for each components (A**2)
        seggamma : numpy array, float (nXsigma_nnum)
            The segment activity coefficients of each segment for each pure component
        seggammam : numpy array, float (1Xsigma_nnum)
            The segment activity coefficients of each segment for the mixture
        gammaSG : numpy array, float (1Xn)
            The Staverman-Guggenheim activity coefficient for each components
        gamma : numpy array, float (1Xn)
            The activity coefficient for each components in the mixtre
        """ 
        segaeff = self.__segaeff
        sigmaprofile = self._get_sigmaprofile()
        sigmaprofilem = self._get_sigmaprofilem()
        sumsigmaarea = self._get_sumsigmaarea()
        seggamma = self._get_seggamma()
        seggammam = self._get_seggammam()
        gammaSG = self._get_gammaSG()
        gamma = np.exp( sumsigmaarea / segaeff * np.sum( sigmaprofile * ( np.log( seggammam ) - np.log( seggamma ) ), axis=1 ) )*\
                gammaSG
        return gamma



class COSMOSAC_MIT(COSMOSACStrategy):

    def set_cp(self, dbname, method):
        """ 
        Parameters
        ----------
        dbname : string
            name of database used in COSMOSAC model from MIT group (in class COSMOSAC_MIT)
            please set one of ['VT', 'UD', 'G09']
        method: string
            version of COSMOSAC model from MIT group (in class COSMOSAC_MIT)
            please set one of ['COSMOSAC-2002','COSMOSAC-2010','COSMOSAC-dsp']
            
        """ 
        self.dbname = dbname
        self.method = method

    def _get_gammam(self):
        """ 
        Parameters
        ----------
        xi : numpy array, float (1Xn) 
            The mole fraction of each components
        ti: float
            temperture (K)
        mixture : list, string (1Xn)
            Component names in the mixture
        gamma : numpy array, float (1Xn)
            The activity coefficient for each components in the mixtre
        """ 
        xi = self.x
        t = self.t
        mixture = self._mixture
        dbname = self.dbname
        method = self.method
        nci = self.nc
        if ( method == 'COSMOSAC-dsp' ) and ( nci >= 3 ):
            lngamma_dsp = np.zeros(nci, dtype=float)
            lngamma_2010 = calc_LNAC(T=t, dbname=dbname, names=mixture, composition=xi, method='COSMOSAC-2010')
            for idx_a, a in enumerate(mixture):
                for idx_b, b in enumerate(mixture[idx_a+1:], start=idx_a+1):
                    mixture_pair = [a, b]
                    xi_pair = [xi[idx_a], xi[idx_b]]
                    lngamma_2010_pair = calc_LNAC(T=t, dbname=dbname, names=mixture_pair, composition=xi_pair, method='COSMOSAC-2010')
                    lngamma_dsp_pair = calc_LNAC(T=t, dbname=dbname, names=mixture_pair, composition=xi_pair, method='COSMOSAC-dsp')
                    lngamma_dsp[idx_a] += ( lngamma_dsp_pair - lngamma_2010_pair )[0]
                    lngamma_dsp[idx_b] += ( lngamma_dsp_pair - lngamma_2010_pair )[1]
            gamma = np.exp(lngamma_2010 + lngamma_dsp)
        else:
            gamma = np.exp(calc_LNAC(T=t, dbname=dbname, names=mixture, composition=xi, method=method))
        return gamma
    
class BinaryModelStrategy(LiquidModelStrategy):
    
    nc = 2
    
    def set_bip(self, a, b, c, d, e, f):
        """
        Parameters
        ----------
        a, b, e, f : numpy array, float nXn matrix 
            binary interaction parameters
        c, d : numpy array, float 1Xn matrix 
            the coordination number c=np.array([Nc1, Nc2])
        bip_type : str
            the type of binary interaction parameters
            abcdef : Aspen's correlation, u/kT=aij+bij/T+eij*lnT+fij*T
            epsilon : u/kT=(Nc/2)*(2*epsilon12-epsilon11-epsilon22)/kT
            
        """
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._e = e
        self._f = f
        self.Nc_array = c.copy()
        self._check_bip()
        self.bip_type = 'abcdef'
        
    def _check_bip(self):
        aij = self._a
        bij = self._b
        eij = self._e
        fij = self._f
        '''
        for mij in (aij, bij, eij, fij):
            if not np.array_equiv(mij, mij.T):
                print(f'The matrix of {mij} is not symmetric.')
        '''
        if not ( ( np.shape(aij) == np.shape(bij) == np.shape(eij) == np.shape(fij) ) and ( np.shape(aij)[0] == np.shape(aij)[1] ) ):
            print('The shape of BIPs are not consistant.')
    
    def set_mixture(self, mixture):
        self._mixture = mixture
    
    def set_epsilon(self, epsilon12dk, epsilon11dk, epsilon22dk, Nc=None, Nc_array=None):
        self.epsilon12dk = epsilon12dk
        self.epsilon11dk = epsilon11dk
        self.epsilon22dk = epsilon22dk
        self.Nc_array = np.array([4, 4])
        if Nc is not None:
            self.Nc_array = np.array([Nc, Nc])
        else:
            if Nc_array is not None:
                self.Nc_array = Nc_array
            else:
                print('Please set the coordination number to Nc or Nc_array.')
        self.bip_type = 'epsilon'
        
    def set_tx(self, t, x):
        self.t = t
        self.x = x
    
    def _get_tau(self):
        ti = self.t
        Nc_arrayi = self.Nc_array
        bip_typei = self.bip_type
        if bip_typei == 'abcdef':
            aij = self._a
            bij = self._b
            eij = self._e
            fij = self._f
            if np.all( eij == 0. ):
                taui = np.exp( - np.sum( ( aij + bij / ti + fij * ti ) / Nc_arrayi ) )
            else:
                taui = np.exp( - np.sum( ( aij + bij / ti + eij * np.log(ti) + fij * ti ) / Nc_arrayi ) )
        elif bip_typei == 'epsilon':
            epsilon12dki = self.epsilon12dk
            epsilon11dki = self.epsilon11dk
            epsilon22dki = self.epsilon22dk
            taui = np.exp( - ( 2 * epsilon12dki - epsilon11dki - epsilon22dki ) / ( 2 * ti ) )
        return taui
    
    def _get_u(self):
        ti = self.t
        bip_typei = self.bip_type
        if bip_typei == 'abcdef':
            aij = self._a
            bij = self._b
            eij = self._e
            fij = self._f
            if np.all( eij == 0. ):
                ui = np.sum( aij + bij / ti + fij * ti )
            else:
                ui = np.sum( aij + bij / ti + eij * np.log( ti ) + fij * ti )
        elif bip_typei == 'epsilon':
            Nc_arrayi = self.Nc_array
            epsilon12dki = self.epsilon12dk
            epsilon11dki = self.epsilon11dk
            epsilon22dki = self.epsilon22dk
            ui = ( Nc_arrayi[0] * ( epsilon12dki - epsilon11dki ) + Nc_arrayi[1] * ( epsilon12dki - epsilon22dki ) ) / ( 2 * ti )
        return ui
    
    def _get_dudt(self):
        ti = self.t
        bip_typei = self.bip_type
        if bip_typei == 'abcdef':
            bij = self._b
            eij = self._e
            fij = self._f
            if np.all( eij == 0. ):
                dudti = np.sum( - bij / ti **2 + fij )
            else:
                dudti = np.sum( - bij / ti **2 + eij / ti + fij )
        elif bip_typei == 'epsilon':
            Nc_arrayi = self.Nc_array
            epsilon12dki = self.epsilon12dk
            epsilon11dki = self.epsilon11dk
            epsilon22dki = self.epsilon22dk
            dudti = - ( Nc_arrayi[0] * ( epsilon12dki - epsilon11dki ) + Nc_arrayi[1] * ( epsilon12dki - epsilon22dki ) ) / ( 2 * ti **2 )
        return dudti
        
    @abstractmethod
    def _get_gammam(self):
        pass
    
    @abstractmethod
    def _get_dmudxa(self):
        pass
    
    @abstractmethod
    def _get_d2mudx2a(self):
        pass
    
    @abstractmethod
    def _get_hexa(self):
        pass
    
    
class Margules(BinaryModelStrategy):

    def _get_gammam(self):
        xi = self.x
        ui = self._get_u()
        gammami = np.exp(  ui * np.flipud( xi ) **2 )
        return gammami
    
    def _get_dmudxa(self):
        xi = self.x
        nci = self.nc
        ui = self._get_u()
        if nci == 2:
            x1, x2 = xi[0], xi[1]
            dmu1dx1 = - 2 * ui * x2 + 1 / x1
            dmu2dx1 = np.nan
            dmu1dx2 = np.nan
            dmu2dx2 = np.nan
            dmudxi = np.array([[dmu1dx1, dmu2dx1], [dmu1dx2, dmu2dx2]])
        else:
            pass
        return dmudxi
    
    def _get_d2mudx2a(self):
        xi = self.x
        nci = self.nc
        ui = self._get_u()
        if nci == 2:
            x1, x2 = xi[0], xi[1]
            d2mu1dx12 = 2 * ui - 1 / x1 **2
            d2mu2dx12 = np.nan
            d2mu1dx22 = np.nan
            d2mu2dx22 = np.nan
            d2mudx2i = np.array([[d2mu1dx12, d2mu2dx12], [d2mu1dx22, d2mu2dx22]])
        else:
            pass
        return d2mudx2i
    
    def _get_hexa(self):
        return self._get_gex
    
class COSMOSACb(BinaryModelStrategy):
        
    def __get_f_inft_square(self):
        xi = self.x
        Nc_arrayi = self.Nc_array
        f_inft_squarei = Nc_arrayi **2 / np.sum( Nc_arrayi * xi )
        return f_inft_squarei
    
    def __get_f_square(self):
        xi = self.x
        Nc_arrayi = self.Nc_array
        taui = self._get_tau()
        f_squarei = ( - taui **2 * ( np.flipud( xi ) * np.flipud( Nc_arrayi ) - xi * Nc_arrayi ) - 2 * xi * Nc_arrayi + \
                    np.sqrt( taui **4 * ( xi[0] * Nc_arrayi[0] - xi[1] * Nc_arrayi[1] ) **2 + 4 * taui **2 * np.prod( xi ) \
                    * np.prod(Nc_arrayi) ) ) / ( 2 * xi **2 * ( taui **2 - 1 ) )
        return f_squarei
        
    def __get_G(self):
        taui = self._get_tau()
        f_squarei = self.__get_f_square()
        Gij = f_squarei[0] / f_squarei[1] * taui
        return Gij
        
    def _get_gammam(self):
        Nc_arrayi = self.Nc_array
        f_inft_squarei = self.__get_f_inft_square()
        f_squarei = self.__get_f_square()
        gammami = ( f_squarei / f_inft_squarei ) **( Nc_arrayi / 2 )
        return gammami
    
    def _get_dmudxa(self):
        print('COSMOSACb model can not analytically calculate first derivatives of chemical potential.')
        
    def _get_d2mudx2a(self):
        print('COSMOSACb model can not analytically calculate second derivatives of chemical potential.')
    
    def _get_hexa(self):
        print('COSMOSACb model can not analytically calculate excess total molar enthalpy.')
        
class QUAC(BinaryModelStrategy):
        
    def __get_z(self):
        xi = self.x
        Nc_arrayi = self.Nc_array
        zi = Nc_arrayi * xi / np.sum( Nc_arrayi * xi )
        return zi
    
    def __get_beta(self):
        zi = self.__get_z()
        taui = self._get_tau()
        betai = np.sqrt( 1 - 4 * ( 1 - taui **( - 2 ) ) *  np.prod( zi ) )
        return betai
        
    def _get_gammam(self):
        xi = self.x
        Nc_arrayi = self.Nc_array
        zi = self.__get_z()
        betai = self.__get_beta()
        z_betai = np.array([zi[0] - zi[1], zi[1] - zi[0]])
        '''
        fail
        prod_zi = np.prod( zi )
        flipud_zi = np.flipud(zi)
        gammami = np.exp( Nc_arrayi / 2 * np.sum( zi * np.log( ( betai + z_betai ) / ( zi * ( 1 + betai ) ) ) ) + np.sum( Nc_arrayi * xi ) / ( 2 * xi ) * \
                         ( prod_zi * np.log( ( betai - z_betai ) * zi / ( ( betai + z_betai ) * flipud_zi ) ) + ( 1 - 2 * zi ) * ( 1 - betai ) / ( 2 * betai ) * \
                          ( 1 - ( 1 + betai ) * np.sum( zi / ( betai + z_betai ) ) ) + 2 * prod_zi * ( zi / ( betai + z_betai[0] ) - flipud_zi / ( betai + z_betai[1] ) ) ) )
        '''
        gammami = ( ( betai + z_betai ) / ( zi * ( 1 + betai ) ) ) **( Nc_arrayi / 2 )
        return gammami
    
    def _get_dmudxa(self):
        print('QUAC model can not analytically calculate first derivatives of chemical potential.')
        
    def _get_d2mudx2a(self):
        print('QUAC model can not analytically calculate second derivatives of chemical potential.')
    
    def _get_hexa(self):
        print('QUAC model can not analytically calculate excess total molar enthalpy.')

class Ising():
    
    def __init__(self, d=2, dt=1.e-4):
        self.num = 1000001
        self.phi = np.linspace(0, np.pi/2, num=self.num, endpoint=True)
        self._epsilon12dk = None
        self._t = None
        self.d = d
        self.dt = dt
    
    @property
    def epsilon12dk(self):
        return self._epsilon12dk
    
    @epsilon12dk.setter
    def epsilon12dk(self, epsilon12dk):
        self._epsilon12dk = epsilon12dk
      
    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, t):
        self._t = t
  
    @property
    def K(self):
        return self.epsilon12dk / self.t
        
    @property
    def Kc(self):
        if self.d == 2:
            Kc = np.log( np.sqrt(2) + 1 ) / 2 # critical point
        elif self.d == 3:
            Kc = 0.2216544
        return Kc
    
    @property
    def kappa(self):
        K = self.K
        return 2 * np.sinh(2 * K) / ( np.cosh(2 * K) ) **2
    
    @property
    def kappa_p(self):
        K = self.K
        return 2 * np.tanh(2 * K) **2 -1
    
    @property
    def K1(self):
        kappa = self.kappa
        num = self.num
        phi = self.phi
        y = 1 / ( ( 1 - ( kappa * np.sin(phi) ) **2 ) **2 ) **( 1 / 4 )
        K1 = np.trapz(y, x=phi, dx=1/num)
        return K1
    
    @property
    def E1(self):
        kappa = self.kappa
        num = self.num
        phi = self.phi
        y = np.sqrt( 1 - ( kappa * np.sin(phi) ) **2 )
        E1 = np.trapz(y, x=phi, dx=1/num)
        return E1
    
    def calc_A(self): #A/NkT: Helmholtz free energy
        if self.d == 2:
            K = self.K
            kappa = self.kappa
            num = self.num
            phi = self.phi
            y = np.log( 1 + ( ( 1 - ( kappa * np.sin(phi) ) **2 ) **2 ) **( 1 / 4 ) )
            integration = np.trapz(y, x=phi, dx=1/num)
            A = - np.log( np.sqrt(2) * np.cosh(2 * K) ) - integration / np.pi
        else:
            A = np.nan
        return A
    
    def calc_At(self): #A/Nk: Helmholtz free energy
        t = self._t
        A = self.calc_A()
        At = A * t
        return At
    
    @property
    def A(self): #A/NkT: Helmholtz free energy
        return self.calc_A()
        
    @property
    def U(self): #U/NkT: internal energy
        if self.d == 2:
            K = self.K
            kappa_p = self.kappa_p
            K1 = self.K1
            U = - K / np.tanh(2 * K) * ( 1 + 2 * kappa_p * K1 / np.pi )
        else:
            U = np.nan
        return U
    
    @property
    def C(self): #C/Nk: Heat capacity
        if self.d == 2:
            K = self.K
            kappa_p = self.kappa_p
            K1 = self.K1
            E1 = self.E1
            C =  2 / np.pi * ( K / np.tanh(2 * K) ) **2 * \
            ( 2 * ( K1 - E1 ) - ( 1 - kappa_p ) * ( np.pi / 2 + kappa_p * K1 ) )
        else:
            C = np.nan
        return C    
    
    @decorator_d2mdt2n(mi=calc_At)
    def calc_d2Atdt2n(self, dt):
        pass
    
    @property
    def Cn(self):
        t = self._t
        dt = self.dt
        d2Atdt2n = self.calc_d2Atdt2n(dt)
        Cn = - t * d2Atdt2n
        return Cn
    
    @property
    def tr(self):
        Kc = self.Kc
        K = self.K
        return Kc / K
    
    @property
    def M(self):
        d = self.d
        tr = self.tr
        K = self.K
        if d == 2:
            M = ( 1 - 1 / np.sinh( 2 * K ) **4 ) **( 1 / 8 ) if tr <= 1 else 0.
            # M = 1.2224 * ( 1 - tr ) **( 1 / 8 )
        elif d == 3:
            # emrpirical expression for 3D Ising model in the range 0.0005 < t < 0.26
            t = 1 - tr
            M = t **0.32694109 * ( 1.6919045 - 0.34357731 * t **0.50842026 - 0.42572366 * t ) if t >= 0 else 0.
        return M
        
def create(kind):
    """
    Create an liquidmodel object.

    Parameters
    ----------
    kind : string
        Type of liquidmodel
    Returns
    -------
    class of liquid model
    """
    kind_list = ['NRTL', 'Wilson', 'COSMOSAC2002e', 'Margules', 'COSMOSACb', 'QUAC', 'COSMOSAC_MIT']
    if kind == kind_list[0]:
        return NRTL()
    elif kind == kind_list[1]:
        return Wilson()
    elif kind == kind_list[2]:
        return COSMOSAC2002e()
    elif kind == kind_list[3]:
        return Margules()
    elif kind == kind_list[4]:
        return COSMOSACb()
    elif kind == kind_list[5]:
        return QUAC()
    elif kind == kind_list[6]:
        return COSMOSAC_MIT()
    else:
        raise ValueError(f'kind must be one of {kind_list}')