from pathlib import Path, PureWindowsPath
import struct, sys, warnings

# From http://stackoverflow.com/a/28952464/1060738

if sys.hexversion < 0x03040000:
    warnings.warn("'{}' module requires python3.4 version or above".format(__file__), ImportWarning)


# doc says class id =
#    00021401-0000-0000-C000-000000000046
# requiredCLSID = b'\x00\x02\x14\x01\x00\x00\x00\x00\xC0\x00\x00\x00\x00\x00\x00\x46'
# Actually Getting:
requiredCLSID   = b'\x01\x14\x02\x00\x00\x00\x00\x00\xC0\x00\x00\x00\x00\x00\x00\x46'  # puzzling

class ShortCutError(RuntimeError):
    pass


class MSShortcut():
    """
    interface to Microsoft Shortcut Objects.  Purpose:
    - I need to be able to get the target from a samba shared on a linux machine
    - Also need to get access from a Windows machine.
    - Need to support several forms of the shortcut, as they seem be created differently depending on the
      creating machine.
    - Included some 'flag' types in external interface to help test differences in shortcut types

    Args:
        scPath (str): path to shortcut

    Limitations:
        - There are some omitted object properties in the implementation.
          Only implemented / tested enough to recover the shortcut target information. Recognized omissions:
          - LinkTargetIDList
          - VolumeId structure (if captured later, should be a separate class object to hold info)
          - Only captured environment block from extra data
        - I don't know how or when some of the shortcut information is used, only captured what I recognized,
          so there may be bugs related to use of the information
        - NO shortcut update support (though might be nice)
        - Implementation requires python 3.4 or greater
        - Tested only with Unicode data on a 64bit little endian machine, did not consider potential endian issues

    Not Debugged:
        - localBasePath - didn't check if parsed correctly or not.
        - commonPathSuffix
        - commonNetworkRelativeLink

    """

    def __init__(self, scPath):
        """
        Parse and keep shortcut properties on creation
        """
        self.scPath = Path(scPath)

        self._clsid = None
        self._lnkFlags = None
        self._lnkInfoFlags = None
        self._localBasePath = None
        self._commonPathSuffix = None
        self._commonNetworkRelativeLink = None
        self._name = None
        self._relativePath = None
        self._workingDir = None
        self._commandLineArgs = None
        self._iconLocation = None
        self._envTarget = None

        self._ParseLnkFile(self.scPath)


    @property
    def clsid(self):
        return self._clsid

    @property
    def lnkFlags(self):
        return self._lnkFlags

    @property
    def lnkInfoFlags(self):
        return self._lnkInfoFlags

    @property
    def localBasePath(self):
        return self._localBasePath

    @property
    def commonPathSuffix(self):
        return self._commonPathSuffix

    @property
    def commonNetworkRelativeLink(self):
        return self._commonNetworkRelativeLink

    @property
    def name(self):
        return self._name

    @property
    def relativePath(self):
        return self._relativePath

    @property
    def workingDir(self):
        return self._workingDir

    @property
    def commandLineArgs(self):
        return self._commandLineArgs

    @property
    def iconLocation(self):
        return self._iconLocation

    @property
    def envTarget(self):
        return self._envTarget


    @property
    def targetPath(self):
        """
        Args:
            woAnchor (bool): remove the anchor (\\server\path or drive:) from returned path.

        Returns:
            a libpath PureWindowsPath object for combined workingDir/relative path
            or the envTarget

        Raises:
            ShortCutError when no target path found in Shortcut
        """
        target = None
        if self.workingDir:
            target = PureWindowsPath(self.workingDir)
            if self.relativePath:
                target = target / PureWindowsPath(self.relativePath)
            else: target = None

        if not target and self.envTarget:
            target = PureWindowsPath(self.envTarget)

        if not target:
            raise ShortCutError("Unable to retrieve target path from MS Shortcut: shortcut = {}"
                               .format(str(self.scPath)))

        return target

    @property
    def targetPathWOAnchor(self):
        tp = self.targetPath
        return tp.relative_to(tp.anchor)




    def _ParseLnkFile(self, lnkPath):
        with lnkPath.open('rb') as f:
            content = f.read()

            # verify size  (4 bytes)
            hdrSize = struct.unpack('I', content[0x00:0x04])[0]
            if hdrSize != 0x4C:
                raise ShortCutError("MS Shortcut HeaderSize = {}, but required to be = {}: shortcut = {}"
                                   .format(hdrSize, 0x4C, str(lnkPath)))

            # verify LinkCLSID id (16 bytes)
            self._clsid = bytes(struct.unpack('B'*16, content[0x04:0x14]))
            if self._clsid != requiredCLSID:
                raise ShortCutError("MS Shortcut ClassID = {}, but required to be = {}: shortcut = {}"
                                   .format(self._clsid, requiredCLSID, str(lnkPath)))

            # read the LinkFlags structure (4 bytes)
            self._lnkFlags = struct.unpack('I', content[0x14:0x18])[0]
            #logger.debug("lnkFlags=0x%0.8x" % self._lnkFlags)
            position = 0x4C

            # if HasLinkTargetIDList bit, then position to skip the stored IDList structure and header
            if (self._lnkFlags & 0x01):
                idListSize = struct.unpack('H', content[position:position+0x02])[0]
                position += idListSize + 2

            # if HasLinkInfo, then process the linkinfo structure
            if (self._lnkFlags & 0x02):
                (linkInfoSize, linkInfoHdrSize, self._linkInfoFlags,
                 volIdOffset, localBasePathOffset,
                 cmnNetRelativeLinkOffset, cmnPathSuffixOffset) = struct.unpack('IIIIIII', content[position:position+28])

                # check for optional offsets
                localBasePathOffsetUnicode = None
                cmnPathSuffixOffsetUnicode = None
                if linkInfoHdrSize >= 0x24:
                    (localBasePathOffsetUnicode, cmnPathSuffixOffsetUnicode) = struct.unpack('II', content[position+28:position+36])

                #logger.debug("0x%0.8X" % linkInfoSize)
                #logger.debug("0x%0.8X" % linkInfoHdrSize)
                #logger.debug("0x%0.8X" % self._linkInfoFlags)
                #logger.debug("0x%0.8X" % volIdOffset)
                #logger.debug("0x%0.8X" % localBasePathOffset)
                #logger.debug("0x%0.8X" % cmnNetRelativeLinkOffset)
                #logger.debug("0x%0.8X" % cmnPathSuffixOffset)
                #logger.debug("0x%0.8X" % localBasePathOffsetUnicode)
                #logger.debug("0x%0.8X" % cmnPathSuffixOffsetUnicode)

                # if info has a localBasePath
                if (self._linkInfoFlags & 0x01):
                    bpPosition = position + localBasePathOffset

                    # not debugged - don't know if this works or not
                    self._localBasePath = UnpackZ('z', content[bpPosition:])[0].decode('ascii')
                    #logger.debug("localBasePath: {}".format(self._localBasePath))

                    if localBasePathOffsetUnicode:
                        bpPosition = position + localBasePathOffsetUnicode
                        self._localBasePath = UnpackUnicodeZ('z', content[bpPosition:])[0]
                        self._localBasePath = self._localBasePath.decode('utf-16')
                        #logger.debug("localBasePathUnicode: {}".format(self._localBasePath))

                # get common Path Suffix
                cmnSuffixPosition = position + cmnPathSuffixOffset
                self._commonPathSuffix = UnpackZ('z', content[cmnSuffixPosition:])[0].decode('ascii')
                #logger.debug("commonPathSuffix: {}".format(self._commonPathSuffix))
                if cmnPathSuffixOffsetUnicode:
                    cmnSuffixPosition = position + cmnPathSuffixOffsetUnicode
                    self._commonPathSuffix = UnpackUnicodeZ('z', content[cmnSuffixPosition:])[0]
                    self._commonPathSuffix = self._commonPathSuffix.decode('utf-16')
                    #logger.debug("commonPathSuffix: {}".format(self._commonPathSuffix))


                # check for CommonNetworkRelativeLink
                if (self._linkInfoFlags & 0x02):
                    relPosition = position + cmnNetRelativeLinkOffset
                    self._commonNetworkRelativeLink = CommonNetworkRelativeLink(content, relPosition)

                position += linkInfoSize

            # If HasName
            if (self._lnkFlags & 0x04):
                (position, self._name) = self.readStringObj(content, position)
                #logger.debug("name: {}".format(self._name))

            # get relative path string
            if (self._lnkFlags & 0x08):
                (position, self._relativePath) = self.readStringObj(content, position)
                #logger.debug("relPath='{}'".format(self._relativePath))

            # get working dir string
            if (self._lnkFlags & 0x10):
                (position, self._workingDir) = self.readStringObj(content, position)
                #logger.debug("workingDir='{}'".format(self._workingDir))

            # get command line arguments
            if (self._lnkFlags & 0x20):
                (position, self._commandLineArgs) = self.readStringObj(content, position)
                #logger.debug("commandLineArgs='{}'".format(self._commandLineArgs))

            # get icon location
            if (self._lnkFlags & 0x40):
                (position, self._iconLocation) = self.readStringObj(content, position)
                #logger.debug("iconLocation='{}'".format(self._iconLocation))

            # look for environment properties
            if (self._lnkFlags & 0x200):
                while True:
                    size = struct.unpack('I', content[position:position+4])[0]
                    #logger.debug("blksize=%d" % size)
                    if size==0: break

                    signature = struct.unpack('I', content[position+4:position+8])[0]
                    #logger.debug("signature=0x%0.8x" % signature)

                    # EnvironmentVariableDataBlock
                    if signature == 0xA0000001:

                        if (self._lnkFlags & 0x80): # unicode
                            self._envTarget = UnpackUnicodeZ('z', content[position+268:])[0]
                            self._envTarget = self._envTarget.decode('utf-16')
                        else:
                            self._envTarget = UnpackZ('z', content[position+8:])[0].decode('ascii')

                        #logger.debug("envTarget='{}'".format(self._envTarget))

                    position += size


    def readStringObj(self, scContent, position):
        """
        returns:
            tuple: (newPosition, string)
        """
        strg = ''
        size = struct.unpack('H', scContent[position:position+2])[0]
        #logger.debug("workingDirSize={}".format(size))
        if (self._lnkFlags & 0x80): # unicode
            size *= 2
            strg = struct.unpack(str(size)+'s', scContent[position+2:position+2+size])[0]
            strg = strg.decode('utf-16')
        else:
            strg = struct.unpack(str(size)+'s', scContent[position+2:position+2+size])[0].decode('ascii')
        #logger.debug("strg='{}'".format(strg))
        position += size + 2 # 2 bytes to account for CountCharacters field

        return (position, strg)




class CommonNetworkRelativeLink():

    def __init__(self, scContent, linkContentPos):

        self._networkProviderType = None
        self._deviceName = None
        self._netName = None

        (linkSize, flags, netNameOffset,
         devNameOffset, self._networkProviderType) = struct.unpack('IIIII', scContent[linkContentPos:linkContentPos+20])

        #logger.debug("netnameOffset = {}".format(netNameOffset))
        if netNameOffset > 0x014:
            (netNameOffsetUnicode, devNameOffsetUnicode) = struct.unpack('II', scContent[linkContentPos+20:linkContentPos+28])
            #logger.debug("netnameOffsetUnicode = {}".format(netNameOffsetUnicode))
            self._netName = UnpackUnicodeZ('z', scContent[linkContentPos+netNameOffsetUnicode:])[0]
            self._netName = self._netName.decode('utf-16')
            self._deviceName = UnpackUnicodeZ('z', scContent[linkContentPos+devNameOffsetUnicode:])[0]
            self._deviceName = self._deviceName.decode('utf-16')
        else:
            self._netName = UnpackZ('z', scContent[linkContentPos+netNameOffset:])[0].decode('ascii')
            self._deviceName = UnpackZ('z', scContent[linkContentPos+devNameOffset:])[0].decode('ascii')

    @property
    def deviceName(self):
        return self._deviceName

    @property
    def netName(self):
        return self._netName

    @property
    def networkProviderType(self):
        return self._networkProviderType



def UnpackZ (fmt, buf) :
    """
    Unpack Null Terminated String
    """
    #logger.debug(bytes(buf))

    while True :
        pos = fmt.find ('z')
        if pos < 0 :
            break
        z_start = struct.calcsize (fmt[:pos])
        z_len = buf[z_start:].find(b'\0')
        #logger.debug(z_len)
        fmt = '%s%dsx%s' % (fmt[:pos], z_len, fmt[pos+1:])
        #logger.debug("fmt='{}', len={}".format(fmt, z_len))
    fmtlen = struct.calcsize(fmt)
    return struct.unpack (fmt, buf[0:fmtlen])


def UnpackUnicodeZ (fmt, buf) :
    """
    Unpack Null Terminated String
    """
    #logger.debug(bytes(buf))
    while True :
        pos = fmt.find ('z')
        if pos < 0 :
            break
        z_start = struct.calcsize (fmt[:pos])
        # look for null bytes by pairs
        z_len = 0
        for i in range(z_start,len(buf),2):
            if buf[i:i+2] == b'\0\0':
                z_len = i-z_start
                break

        fmt = '%s%dsxx%s' % (fmt[:pos], z_len, fmt[pos+1:])
       # logger.debug("fmt='{}', len={}".format(fmt, z_len))
    fmtlen = struct.calcsize(fmt)
    return struct.unpack (fmt, buf[0:fmtlen])